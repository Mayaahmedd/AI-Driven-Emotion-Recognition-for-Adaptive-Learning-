# Read all remaining test reports in one command
for exp in \
  "RESULTS/exp31_resnet_bilstm_attention_balancedcsv/checkpoints/exp31_resnet_bilstm_attention_balancedcsv" \
  "RESULTS/exp34_multilabel_resnet_bilstm_attention_oversampled/checkpoints/exp34_multilabel_resnet_bilstm_attention_oversampled" \
  "RESULTS/exp35_shufflenet_bilstm_attention_oversampled/checkpoints/exp35_shufflenet_bilstm_attention_oversampled" \
  "RESULTS/exp36_resnet_bilstm2_attention_24frames/checkpoints/exp36_resnet_bilstm2_attention_24frames" \
  "RESULTS/exp37_shufflenet_se_bilstm2_attention_24frames/checkpoints/exp37_shufflenet_se_bilstm2_attention_24frames"; do
  echo "====== $exp ======"
  [ -f "$exp/test_report.json" ] && cat "$exp/test_report.json" || echo "NO TEST REPORT"
  echo ""
done

# SVM reports are .txt not .json
echo "====== HOG+LBP SVM ======"
cat RESULTS/traditional_ml_hog_lbp_svm/test_report.txt 2>/dev/null || \
  find RESULTS/traditional_ml_hog_lbp_svm -type f | sort

echo ""
echo "====== ResNet50 SVM ======"
cat RESULTS/resnet50_features_svm/test_report.txt 2>/dev/null || \
  find RESULTS/resnet50_features_svm -type f | sort

# Exp 16 full train log to get per-label precision and recall
echo ""
echo "====== EXP 16 FULL TRAIN LOG ======"
cat RESULTS/exp16_save/train_log.json