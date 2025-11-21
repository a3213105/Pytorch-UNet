start_core=32
for scale in 0.5 1.0
do
  for core in 1 2 32
  do
    echo "core $core, scale $scale"
    core_end=$((start_core + core - 1))
    echo "numactl -C ${start_core}-${core_end} python predict_ov.py -i ../1.jpg -m ../UNet-models/unet_carvana_scale${scale}_epoch2.pth -s ${scale} -n $core"
    logfile=/tmp/unet_${core_end}_${scale}_${core}.log
    numactl -C ${start_core}-${core_end} python predict_ov.py -i ../1.jpg -m ../UNet-models/unet_carvana_scale${scale}_epoch2.pth -s ${scale} -n $core | grep FPS > ${logfile}
    torch_f32=$(grep "Torch_F32" ${logfile}| awk '{print $11}')
    echo "Torch_F32 FPS: $torch_f32"

    torch_bf16=$(grep "Torch_BF16" ${logfile} | awk '{print $11}')
    ratio=$(awk "BEGIN {printf \"%.4f\", $torch_bf16 / $torch_f32}")
    echo "Torch_BF16 FPS: $torch_bf16, ${ratio}"

    value=$(grep "OV.*F32" ${logfile} | awk '{print $11, $1}' | sort -nr | head -1)
    ov_f32_max=$(echo ${value} | awk '{print $1}')
    ov_f32_type=$(echo ${value} | awk '{print $2}')
    ratio=$(awk "BEGIN {printf \"%.4f\", $ov_f32_max / $torch_f32}")
    echo "OV F32 Max FPS: $ov_f32_max (Type: $ov_f32_type), ${ratio}"

    value=$(grep "OV.*BF16" ${logfile} | awk '{print $11, $1}' | sort -nr | head -1)
    ov_bf16_max=$(echo ${value} | awk '{print $1}')
    ov_bf16_type=$(echo ${value} | awk '{print $2}')
    ratio=$(awk "BEGIN {printf \"%.4f\", $ov_bf16_max / $torch_f32}")
    echo "OV BF16 Max FPS: $ov_bf16_max (Type: $ov_bf16_type), ${ratio}"

    value=$(grep "OV.*_F16" ${logfile} | awk '{print $11, $1}' | sort -nr | head -1)
    ov_f16_max=$(echo ${value} | awk '{print $1}')
    ov_f16_type=$(echo ${value} | awk '{print $2}')
    ratio=$(awk "BEGIN {printf \"%.4f\", $ov_f16_max / $torch_f32}")
    echo "OV F16 Max FPS: $ov_f16_max (Type: $ov_f16_type), ${ratio}"
  done
done
