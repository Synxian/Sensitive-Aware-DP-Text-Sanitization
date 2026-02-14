python run_selective_dp.py --data_dir ./datasets/{dataset_name}/ --method normal --task {dataset_name} --epsilon 1 --s_epsilon 0.5 && /
python run_selective_dp.py --data_dir ./datasets/{dataset_name}/ --method normal --task {dataset_name} --epsilon 2 --s_epsilon 1 && /
python run_selective_dp.py --data_dir ./datasets/{dataset_name}/ --method normal --task {dataset_name} --epsilon 4 --s_epsilon 2 && /
python run_selective_dp.py --data_dir ./datasets/{dataset_name}/ --method normal --task {dataset_name} --epsilon 8 --s_epsilon 4 && /
python run_selective_dp.py --data_dir ./datasets/{dataset_name}/ --method normal --task {dataset_name} --epsilon 16 --s_epsilon 8 && /
python run_selective_dp.py --data_dir ./datasets/{dataset_name}/ --method normal --task {dataset_name} --epsilon 32 --s_epsilon 16 && /
python run_selective_dp.py --data_dir ./datasets/{dataset_name}/ --method plus --task {dataset_name} --epsilon 1 --s_epsilon 0.5 && /
python run_selective_dp.py --data_dir ./datasets/{dataset_name}/ --method plus --task {dataset_name} --epsilon 2 --s_epsilon 1 && /
python run_selective_dp.py --data_dir ./datasets/{dataset_name}/ --method plus --task {dataset_name} --epsilon 4 --s_epsilon 2 && /
python run_selective_dp.py --data_dir ./datasets/{dataset_name}/ --method plus --task {dataset_name} --epsilon 8 --s_epsilon 4 && /
python run_selective_dp.py --data_dir ./datasets/{dataset_name}/ --method plus --task {dataset_name} --epsilon 16 --s_epsilon 8 && /
python run_selective_dp.py --data_dir ./datasets/{dataset_name}/ --method plus --task {dataset_name} --epsilon 32 --s_epsilon 16