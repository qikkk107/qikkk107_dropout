各文件夹说明：
1dropout_Initial data processing -----辍学初始数据处理，包含对数据清洗、特征工程造价、初步分类预测
2dropout_MachineLearning----基于机器学习的辍学预测模型建构，在此之前还对数据进行了不均衡处理
3dropout_DeepLearning----基于深度学习的辍学预测模型建构，使用的使不均衡处理后的数据
4dropout_FeatureImportance----关键特征提取，使用评估出的较好性能的随机森林预测模型绘制重要性特征图 
final_feature_all.csv----初始数据处理后生成的CSV文件，后续机器学习、深度学习、关键特征提取均基于此数据集                                    