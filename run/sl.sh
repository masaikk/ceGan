sleep 6000

python ./run_cegan.py --temp 80 --mu_temp "exp sigmoid" --fn_mu_temp "exp sigmoid"
python ./run_cegan.py --temp 15 --mu_temp "exp sigmoid" --fn_mu_temp "exp sigmoid"