#conda activate textgan;
#python ./run_cegan.py;
#python ./run_cegan.py --mu_temp log;
#python ./run_cegan.py --mu_temp exp;
#python ./run_cegan.py --mu_temp exp;

#python ./run_cegan.py --temp 50 --mu_temp "exp log" --fn_mu_temp "exp";
#python ./run_cegan.py --temp 50 --mu_temp "exp";
#python ./run_cegan.py --temp 25 --mu_temp "exp";
#python ./run_cegan.py --temp 10 --mu_temp "exp";


#python ./run_cegan.py --temp 100 --mu_temp exp --fn_mu_temp "exp sigmoid quad"
#python ./run_cegan.py --temp 200 --mu_temp exp --fn_mu_temp "exp sigmoid quad"
#python ./run_cegan.py --temp 150 --mu_temp "exp sigmoid quad" --fn_mu_temp exp

#python ./run_cegan.py --temp 100 --mu_temp exp --fn_mu_temp exp
#python ./run_cegan.py --temp 25 --mu_temp exp --fn_mu_temp exp
#python ./run_cegan.py --temp 100 --mu_temp "exp sigmoid" --fn_mu_temp exp

#python ./run_cegan.py --temp 100 --mu_temp "exp sigmoid" --fn_mu_temp "exp sigmoid"
#python ./run_cegan.py --temp 50 --mu_temp "exp sigmoid" --fn_mu_temp "exp sigmoid"

python ./run_cegan.py --temp 80 --mu_temp "exp sigmoid" --fn_mu_temp "exp sigmoid"
python ./run_cegan.py --temp 15 --mu_temp "exp sigmoid" --fn_mu_temp "exp sigmoid"