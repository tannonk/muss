# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copyreg import pickle
import sys
from pathlib import Path

import faiss
from tqdm import tqdm

from muss.utils.submitit import get_executor
from muss.utils.helpers import get_file_hash, get_files_hash, log_action, yield_lines
from muss.resources.paths import get_dataset_dir
from muss.laser import get_laser_embeddings
from muss.mining.preprocessing import (
    get_subshard_paths,
    get_sentences_paths,
    sentence_tokenize_subshard,
    split_ccnet_shard,
    create_base_index,
    get_index_name,
    pickle_train_sentences,
)
from muss.mining.nn_search import (
    get_cache_dir,
    get_results_path,
    compute_and_save_nn_batched,
    get_paraphrase_pairs,
    get_pairs_path,
    compute_and_save_simplification_pairs,
    get_index_path,
    compute_and_save_embeddings,
    get_filter_string_representation,
    combine_simplifications_in_dataset,
    get_simplification_pairs_paths,
    wrap_up_paraphrases
)
from muss.mining.filtering import SimplicityScorer

# ccnet_dir = Path(
#     input(
#         'Please download the CCNet corpus from https://github.com/facebookresearch/cc_net and enter the path to the downloaded data: '
#     )
# )

# ccnet_dir = Path('/scratch/tkew/cc_net/test_data/mined/2019-09') # test_data
ccnet_dir = Path('/scratch/tkew/cc_net/data/mined_split/2019-09') # s3it
language = sys.argv[1] # input('What language do you want to process? (en/fr/es/de): ')
# cluster = 'local'
cluster = 'slurm'
# dataset_dir = get_dataset_dir('test_data') / language
dataset_dir = get_dataset_dir('uts') / language
print('dataset dir:', dataset_dir)
# For large jobs only
# slurm_partition = 'dev,scavenge'
gpu_partition = 'volta'
cpu_partition = 'generic'
slurm_gres="gpu:1"
# slurm_array_parallelism = 256
slurm_array_parallelism = 32
mem_gb=32
cpus_per_task=8


# Split CCNet shards into subshards
with log_action('Splitting CCNet shards into smaller subshards'):
    # We need to split each shard even more for the LASER embeddings to fit in memory
    n_shards = {  # Number of shards to take for each languages for ~1B sentences
        'de': 1599, # requires language model!! https://github.com/facebookresearch/muss/issues/20#issuecomment-1031474509
        'en': 340, #15,
        'fr': 23, #25,
        'es': 13,  # We would need about 20 shards for 1B sentences, but there are only 13
    }[language]
    # ccnet_filepaths = [ccnet_dir / f'{language}_head_{i:04d}.json.gz' for i in range(n_shards)] # test_data format
    ccnet_filepaths = [ccnet_dir / f'{i:04d}' / f'{language}_head.json.gz' for i in range(n_shards)] # mined data format
    raw_original_dir = dataset_dir / 'raw_original'
    raw_original_dir.mkdir(exist_ok=True, parents=True)
    output_dirs = [raw_original_dir / f'{language}_head_{i:04d}' for i in range(n_shards)]
    # print(output_dirs)
    n_docs_per_file = 50000
    # executor = get_executor(cluster=cluster, slurm_partition='dev', timeout_min=1 * 60, slurm_array_parallelism=16)
    executor = get_executor(
        cluster=cluster, 
        slurm_partition=cpu_partition, 
        timeout_min=1 * 60, # NOTE: shouldn't take more than 5mins per job
        slurm_array_parallelism=slurm_array_parallelism, 
        slurm_gres=None, 
        cpus_per_task=1, 
        mem_gb=4
        )
    # import pdb;pdb.set_trace()
    print('[1] Split executor:', executor.parameters)
    jobs = []
    with executor.batch():
        for ccnet_filepath, output_dir in zip(ccnet_filepaths, output_dirs):
            if output_dir.exists():
                continue
            job = executor.submit(split_ccnet_shard, ccnet_filepath, output_dir, n_docs_per_file)
            jobs.append(job)
    print([job.job_id for job in jobs])
    [job.result() for job in tqdm(jobs)]  # Wait for the jobs to finish

# Sentence tokenization
with log_action('Tokenizing sentences'):
    executor = get_executor(
        cluster=cluster,
        slurm_partition=cpu_partition,
        timeout_min=1 * 60, # NOTE: shouldn't take more than 10mins per job
        slurm_array_parallelism=slurm_array_parallelism,
        slurm_gres=None, 
        cpus_per_task=1, 
        mem_gb=16, # NOTE: size of dewiki lm is ~ 9 GB!
    )
    print('[2] Tokenizer executor:', executor.parameters)
    subshard_paths = get_subshard_paths(raw_original_dir)
    jobs = []
    with executor.batch():
        for i, subshard_path in enumerate(subshard_paths):
            sentences_path = dataset_dir / 'sentences' / f'{i:06d}.txt.gz'
            if sentences_path.exists():
                continue
            sentences_path.parent.mkdir(exist_ok=True, parents=True)
            # Should take a bit less than 10 minutes each
            job = executor.submit(sentence_tokenize_subshard, subshard_path, sentences_path, language)
            jobs.append(job)
    print([job.job_id for job in jobs])
    [job.result() for job in tqdm(jobs)]

embeddings_type_name = f'laser_{language}'
get_embeddings = lambda sentences: get_laser_embeddings(
    sentences, max_tokens=3000, language=language, n_encoding_jobs=10
)  # noqa: E731


# import pdb;pdb.set_trace()
with log_action('Collecting train sentences'):

    train_sentences = dataset_dir / 'train_sentences.pkl'
    if not train_sentences.exists():

        executor = get_executor(
            cluster=cluster,
            slurm_partition=cpu_partition,
            timeout_min=4 * 60,
            slurm_array_parallelism=slurm_array_parallelism,
            slurm_gres=None,
            cpus_per_task=1, 
            mem_gb=mem_gb,
        )
        print('[3] Index executor:', executor.parameters)

        jobs = []
        with executor.batch():
            job = executor.submit(pickle_train_sentences, dataset_dir, train_sentences)
            jobs.append(job)

        print([job.job_id for job in jobs])
        train_sentences = [job.result() for job in tqdm(jobs)][0]
    else:
        print(f'Found exisiting pickled training sentences at {train_sentences}')


# Create base index
with log_action('Creating base index'):
    base_index_dir = dataset_dir / f'base_indexes/'
    print('BASE INDEX DIR', base_index_dir)
    base_index_dir.mkdir(exist_ok=True, parents=True)
    # skip this step if base index already exisits. 
    # saves waiting for job to start to find out whether or not we can skip this step.
    index_name = get_index_name()
    index_prefix = f'{index_name.replace(",", "_").lower()}_metric{faiss.METRIC_L2}'
    base_index_path = base_index_dir / f'{index_prefix}.faiss_index'

    if base_index_path.exists():
        print('Base index exists. Skipping...')
    else:
        # This can be very long (i.e. 46971.065 secs ~ 14 hours), e.g.
        # Computing embeddings took approx 7882.18s (on 10000000 en sents)
        # Training index took approx 39061.53s.
	# for DE 1599 segments, took 101882 srcs ~ 28 hours            
        # execute with submitit
        executor = get_executor(
            cluster=cluster,
            slurm_partition=gpu_partition,
            timeout_min=32 * 60,
            slurm_array_parallelism=slurm_array_parallelism,
            slurm_gres=slurm_gres,
            cpus_per_task=1, 
            mem_gb=mem_gb,
        )

        print('[4] Index executor:', executor.parameters)

        jobs = [] # only a single job, but copied template from others
        with executor.batch():
            job = executor.submit(create_base_index, train_sentences, get_index_name(), get_embeddings, faiss.METRIC_L2, base_index_dir)
            jobs.append(job)

        print([job.job_id for job in jobs])
        base_index_path = [job.result() for job in tqdm(jobs)][0]
    
print('BASE INDEX PATH:', base_index_path)


# Compute embeddings
with log_action('Computing embeddings'):
    cache_dir = get_cache_dir(dataset_dir) / embeddings_type_name
    print('CACHE DIR', cache_dir)
    indexes_dir = cache_dir / 'indexes' / f'base-index-{get_file_hash(base_index_path)}'
    indexes_dir.mkdir(exist_ok=True, parents=True)
    db_sentences_paths = get_sentences_paths(dataset_dir)
    query_sentences_paths = db_sentences_paths
    executor = get_executor(
        cluster=cluster,
        slurm_partition=gpu_partition,
        timeout_min=6 * 60,
        slurm_array_parallelism=slurm_array_parallelism,
        slurm_gres=slurm_gres,
        cpus_per_task=cpus_per_task, 
        mem_gb=mem_gb,
    )
    print('[5] Compute embeddings executor:', executor.parameters)
    jobs = []
    with executor.batch():
        for sentences_path in set(query_sentences_paths + db_sentences_paths):
            if get_index_path(sentences_path, indexes_dir).exists():
                continue
            # Should take about 30 minutes each
            job = executor.submit(
                compute_and_save_embeddings, sentences_path, base_index_path, get_embeddings, indexes_dir=indexes_dir
            )
            jobs.append(job)
    print([job.job_id for job in jobs])
    [job.result() for job in tqdm(jobs)]

# Mine the paraphrases
with log_action('Mining paraphrases'):
    nn_search_results_dir = cache_dir / 'nn_search_results'
    nn_search_results_dir.mkdir(exist_ok=True, parents=True)
    topk = 8
    nprobe = 16
    executor = get_executor(
        cluster=cluster,
        slurm_partition=gpu_partition,
        timeout_min=6 * 60,
        slurm_array_parallelism=slurm_array_parallelism,
        slurm_gres=slurm_gres,
        cpus_per_task=cpus_per_task, 
        mem_gb=mem_gb,
    )
    print('[6] Mining executor:', executor.parameters)
    jobs = []
    # Run NN search query file by query file
    # NOTE: jobs are executed in parallel. slurm_array_parallelism is important here for limiting the number of job that can be submitted!
    with executor.batch():
        for query_sentences_path in tqdm(query_sentences_paths, desc='submitting queries'):
            if get_results_path(query_sentences_path, db_sentences_paths, topk, nprobe, nn_search_results_dir).exists():
                continue
            # Should take about ~1h30 each
            # https://github.com/facebookresearch/muss/issues/32#issuecomment-1035031775
            job = executor.submit(
                compute_and_save_nn_batched,
                query_sentences_path,
                db_sentences_paths,
                topk,
                nprobe,
                indexes_dir,
                nn_search_results_dir,
                delete_intermediary=True,
            )
            # job = compute_and_save_nn_batched(
            #     query_sentences_path,
            #     db_sentences_paths,
            #     topk,
            #     nprobe,
            #     indexes_dir,
            #     nn_search_results_dir,
            #     delete_intermediary=True,
            # )
            # import pdb;pdb.set_trace()
            jobs.append(job)
    print([job.job_id for job in jobs])
    [job.result() for job in tqdm(jobs)]

# Filter candidate paraphrases
with log_action('Filtering candidate paraphrases'):
    pairs_dir = cache_dir / 'pairs'
    pairs_dir.mkdir(exist_ok=True, parents=True)
    filter_kwargs = {
        'density': 0.6,
        'distance': 0.05,
        'levenshtein': 0.2,
        'simplicity': 0.0,
        'filter_ne': False,
    }  # Best for paraphrases
    jobs = []
    paraphrase_pairs = []
    i = 0
    is_simpler = lambda pair: True  # noqa: E731
    # Only used when mining simplifications
    if filter_kwargs.get('simplicity', 0) > 0:
        while len(paraphrase_pairs) < 10000:
            paraphrase_pairs.extend(
                get_paraphrase_pairs(
                    query_sentences_paths[i],
                    db_sentences_paths,
                    base_index_path,
                    get_embeddings,
                    cache_dir,
                    topk,
                    nprobe,
                    filter_kwargs,
                )
            )
            i += 1
        simplicity_scorer = SimplicityScorer(language=language)
        simplicity_scorer.fit(paraphrase_pairs)
        is_simpler = lambda pair: simplicity_scorer.score(*pair) > filter_kwargs['simplicity']  # noqa: E731
    executor = get_executor(
        cluster=cluster,
        slurm_partition=gpu_partition,
        timeout_min=4 * 60,
        slurm_array_parallelism=slurm_array_parallelism,
        slurm_gres=slurm_gres,
        cpus_per_task=cpus_per_task, 
        mem_gb=mem_gb,
    )
    print('[7] Filtering executor:', executor.parameters)
    with executor.batch():
        for query_sentences_path in tqdm(query_sentences_paths, desc='query'):
            simplification_pairs_path = get_pairs_path(
                query_sentences_path, db_sentences_paths, topk, nprobe, filter_kwargs, pairs_dir
            )
            if simplification_pairs_path.exists():
                continue
            # Should take ~10 minutes
            job = executor.submit(
                compute_and_save_simplification_pairs,
                query_sentences_path=query_sentences_path,
                db_sentences_paths=db_sentences_paths,
                base_index_path=base_index_path,
                cache_dir=cache_dir,
                pairs_dir=pairs_dir,
                get_embeddings=get_embeddings,
                topk=topk,
                nprobe=nprobe,
                language=language,
                filter_kwargs=filter_kwargs,
                is_simpler=is_simpler,
            )
            jobs.append(job)
    print([job.job_id for job in jobs])
    [job.result() for job in tqdm(jobs)]


with log_action('Wrapping up paraphrases'):
    # NOTE: slurm execution not yet tested
    executor = get_executor(
        cluster=cluster,
        slurm_partition=cpu_partition,
        timeout_min=1 * 60,
        slurm_array_parallelism=slurm_array_parallelism,
        slurm_gres=None,
        cpus_per_task=1, 
        mem_gb=mem_gb,
    )
    print('[3] Wrap-up executor:', executor.parameters)

    jobs = []
    with executor.batch():
        job = executor.submit(
            wrap_up_paraphrases,
            query_sentences_paths=query_sentences_paths, 
            db_sentences_paths=db_sentences_paths, 
            topk=topk, 
            nprobe=nprobe, 
            filter_kwargs=filter_kwargs, 
            pairs_dir=pairs_dir, 
            language=language
        )
        job.append(job)
    print([job.job_id for job in jobs])
    [job.result() for job in tqdm(jobs)]

print('Done!')