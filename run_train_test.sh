seed=(0 1 2 3 4 5 6 7)

for s in "${seed[@]}"
do
    dataset="seed${s}_MER24_AV"
    python train.py --seed="$s" --dataset="$dataset" --model="AV_Base_v3" --save_model
    load_key="/mnt/public/gxj_2/EmoNet_2B/saved/seed${s}_MER24_AV/AV_Base_v3/test/"
    python test.py --dataset="MER24-test_AV" --model="AV_Base_v3" \
                --load_key="$load_key" \
                --save_name="seed${s}"
done