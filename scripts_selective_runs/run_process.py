import subprocess
import argparse

task_args = {
    'meddocan':[
        '--lang es',
        '--word_embedding_path embeddings/spanish/spanish_embed.vec',
        '--sensitive_words_file_path selective_output/sensitive_mapping/meddocan_sensitive_words_finetuned.json',
        '--data_dir datasets/meddocan/',
        '--task meddocan'
        ],
    'i2b2': [
        '--lang en',
        '--word_embedding_path embeddings/english/glove.840B.300d.txt',
        '--sensitive_words_file_path selective_output/sensitive_mapping/flair_0.6_i2b2.json',
        '--data_dir datasets/i2b2/',
        '--task i2b2'
    ],
    'cwlc':[
        '--lang es',
        '--word_embedding_path embeddings/spanish/spanish_embed.vec',
        '--sensitive_words_file_path selective_output/sensitive_mapping/flair_0.6_cwlc.json',
        '--data_dir datasets/cwlc/',
        '--task cwlc'
    ],
    'mimic' : [
        '--lang en',
        '--word_embedding_path embeddings/english/glove.840B.300d.txt',
        '--sensitive_words_file_path selective_output/sensitive_mapping/flair_0.6_mimic.json',
        '--data_dir datasets/mimic/',
        '--task mimic'
    ]
}

common_args = ['--threads 8']

methods = ['normal', 'plus', 'santext']
p_values = [0.2, 0.5, 0.7, 0.9]
def main():
    parser = argparse.ArgumentParser(description="Run SASTDP")
    parser.add_argument('--task', required=True)
    parser.add_argument('--epsilon_s', type=float, required=True)
    parser.add_argument('--epsilon_n', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    runs = []
    for method in methods:
        specific_args = [
            f'--method {method}',
            f'--epsilon {args.epsilon_n}',
            f'--s_epsilon {args.epsilon_s}',
            f'--seed {args.seed}'
        ]
        if method == 'plus':
            for p in p_values:
                specific_args.append(f'--p {p}')
                runs.append(f"python run_selective_dp.py {' '.join(common_args)} {' '.join(task_args[args.task])} {' '.join(specific_args)}")
        else:
            runs.append(f"python run_selective_dp.py {' '.join(common_args)} {' '.join(task_args[args.task])} {' '.join(specific_args)}")
    cmd =' && '.join(runs)
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    main()
