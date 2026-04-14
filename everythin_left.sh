python run_selective_dp.py --threads 8 --data_dir ./datasets/i2b2/ --method santext --task i2b2 --epsilon 1 && /
python run_selective_dp.py --threads 8 --data_dir ./datasets/i2b2/ --method santext --task i2b2 --epsilon 2 && /
python run_selective_dp.py --threads 8 --data_dir ./datasets/i2b2/ --method santext --task i2b2 --epsilon 4 && /
python run_selective_dp.py --threads 8 --data_dir ./datasets/i2b2/ --method santext --task i2b2 --epsilon 8 && /
python run_selective_dp.py --threads 8 --data_dir ./datasets/i2b2/ --method santext --task i2b2 --epsilon 16 && /
python run_selective_dp.py --threads 8 --data_dir ./datasets/i2b2/ --method santext --task i2b2 --epsilon 32 && /
python run_selective_dp.py --threads 8 --data_dir ./datasets/mimic/ --method santext --task mimic --epsilon 1 && /
python run_selective_dp.py --threads 8 --data_dir ./datasets/mimic/ --method santext --task mimic --epsilon 2 && /
python run_selective_dp.py --threads 8 --data_dir ./datasets/mimic/ --method santext --task mimic --epsilon 4 && /
python run_selective_dp.py --threads 8 --data_dir ./datasets/mimic/ --method santext --task mimic --epsilon 8 && /
python run_selective_dp.py --threads 8 --data_dir ./datasets/mimic/ --method santext --task mimic --epsilon 16 && /
python run_selective_dp.py --threads 8 --data_dir ./datasets/mimic/ --method santext --task mimic --epsilon 32 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --data_dir ./datasets/meddocan/ --method santext --task meddocan --epsilon 1 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --data_dir ./datasets/meddocan/ --method santext --task meddocan --epsilon 2 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --data_dir ./datasets/meddocan/ --method santext --task meddocan --epsilon 4 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --data_dir ./datasets/meddocan/ --method santext --task meddocan --epsilon 8 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --data_dir ./datasets/meddocan/ --method santext --task meddocan --epsilon 16 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --data_dir ./datasets/meddocan/ --method santext --task meddocan --epsilon 32 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --sensitive_words_file_path ./selective_output/sensitive_mapping/0.6_meddocan.json --data_dir ./datasets/meddocan/ --method normal --task meddocan --epsilon 1 --s_epsilon 0.5 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --sensitive_words_file_path ./selective_output/sensitive_mapping/0.6_meddocan.json --data_dir ./datasets/meddocan/ --method normal --task meddocan --epsilon 2 --s_epsilon 1 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --sensitive_words_file_path ./selective_output/sensitive_mapping/0.6_meddocan.json --data_dir ./datasets/meddocan/ --method normal --task meddocan --epsilon 4 --s_epsilon 2 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --sensitive_words_file_path ./selective_output/sensitive_mapping/0.6_meddocan.json --data_dir ./datasets/meddocan/ --method normal --task meddocan --epsilon 8 --s_epsilon 4 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --sensitive_words_file_path ./selective_output/sensitive_mapping/0.6_meddocan.json --data_dir ./datasets/meddocan/ --method normal --task meddocan --epsilon 16 --s_epsilon 8 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --sensitive_words_file_path ./selective_output/sensitive_mapping/0.6_meddocan.json --data_dir ./datasets/meddocan/ --method normal --task meddocan --epsilon 32 --s_epsilon 16 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --sensitive_words_file_path ./selective_output/sensitive_mapping/0.6_meddocan.json --data_dir ./datasets/meddocan/ --method plus --task meddocan --epsilon 1 --s_epsilon 0.5 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --sensitive_words_file_path ./selective_output/sensitive_mapping/0.6_meddocan.json --data_dir ./datasets/meddocan/ --method plus --task meddocan --epsilon 2 --s_epsilon 1 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --sensitive_words_file_path ./selective_output/sensitive_mapping/0.6_meddocan.json --data_dir ./datasets/meddocan/ --method plus --task meddocan --epsilon 4 --s_epsilon 2 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --sensitive_words_file_path ./selective_output/sensitive_mapping/0.6_meddocan.json --data_dir ./datasets/meddocan/ --method plus --task meddocan --epsilon 8 --s_epsilon 4 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --sensitive_words_file_path ./selective_output/sensitive_mapping/0.6_meddocan.json --data_dir ./datasets/meddocan/ --method plus --task meddocan --epsilon 16 --s_epsilon 8 && /
python run_selective_dp.py --language es --word_embedding_path embeddings/spanish/glove-esp.vec --threads 8 --sensitive_words_file_path ./selective_output/sensitive_mapping/0.6_meddocan.json --data_dir ./datasets/meddocan/ --method plus --task meddocan --epsilon 32 --s_epsilon 16
