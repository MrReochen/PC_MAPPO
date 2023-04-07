#!/bin/bash
declare -A predict_dims
declare -A adaptive_ts
declare -A predictor_paths
ARGS=`getopt -o m:c:p:s:w::a::r:: -l map:,cuda:,purpose:,seed:,window::,alg::,run:: -n 'test.sh' -- "$@"`
eval set -- "${ARGS}"

while true
do
    case "$1" in
        -m|--map)
            IFS=',' read -r -a maps <<< "$2";
            shift 2
            ;;
        -c|--cuda)
            IFS=',' read -r -a cuda <<< "$2";
            shift 2
            ;;
        -p|--purpose)
            purpose="$2";
            shift 2
            ;;
        -s|--seed)
            case "$2" in
                "")
                    seed=1;
                    shift 2  
                    ;;
                *)
                    seed="$2";
                    shift 2;
                    ;;
            esac
            ;;
        -w|--window)
            case "$2" in
                "")
                    parallel_windows=1;
                    shift 2  
                    ;;
                *)
                    parallel_windows="$2";
                    shift 2;
                    ;;
            esac
            ;;
        -a|--alg)
            case "$2" in
                "")
                    alg="con_mappo";
                    shift 2  
                    ;;
                *)
                    alg="$2";
                    shift 2;
                    ;;
            esac
            ;;
        -r|--run)
            case "$2" in
                "")
                    run="con_mappo";
                    shift 2  
                    ;;
                *)
                    run="$2";
                    shift 2;
                    ;;
            esac
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done


# predict_dims=(["3s5z_vs_3s6z"]=128 ["MMM2"]=32)
# adaptive_ts=(["3s5z_vs_3s6z"]=10 ["MMM2"]=1)
predictor_paths=(\
    ["3s5z_vs_3s6z"]="results/ada_full/3s5z_vs_3s6z/con_mappo_ada/con_mappo_ada/2021-12-27-20-43-20/models/7351" \
    ["MMM2"]="results/ada_full/MMM2/con_mappo_ada/con_mappo_ada/2021-12-27-09-59-20/models/6943" \
    ["6h_vs_8z"]="results/run_pre/6h_vs_8z/con_mappo/con_mappo/2021-12-29-23-17-07/models/4000" \
    ["corridor"]="results/run_pre/corridor/con_mappo/con_mappo/2021-12-29-23-17-17/models/1561" \
    )

tmux new-session -s $run-$purpose -n base_bash -d
for w in $(seq $(($parallel_windows)))
do
    tmux new-window -n "Window-${w}" -t $run-$purpose
done

index=0
for map in ${maps[@]}
do
    for s in $(seq $(($seed)))
    do
        params="seed:${s}"

        if [ ${predictor_paths[${map}]} ] ; then
            params="${params}+predictor_path:${predictor_paths[${map}]}"
        fi
        params="${params}+purpose:${purpose}"
        cmd="CUDA_VISIBLE_DEVICES=${cuda[${index} % ${#cuda[@]}]} python3 main.py --alg=${alg} --run=${run} --map_name=${map} --params=${params}"
        tmux send-keys -t $run:$[(${index} % $parallel_windows)+1] "$cmd" C-m
        index=$[${index}+1]
        sleep 2s
    done
done