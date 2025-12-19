# Warm start configurations for FLAML
# Absolute top N configurations (by performance)
# Auto-generated from optimization logs

warm_start_configs = {
    # Top 5 configurations for catboost
    'catboost': [
        {"early_stopping_rounds":10,"learning_rate":0.14189952377559728,"n_estimators":8192,"FLAML_sample_size":49659},  # Rank 1: metric=0.315344
        {"early_stopping_rounds":11,"learning_rate":0.1530902242854414,"n_estimators":8192,"FLAML_sample_size":10000},  # Rank 2: metric=0.316097
        {"early_stopping_rounds":12,"learning_rate":0.09541333025917802,"n_estimators":8192,"FLAML_sample_size":49659},  # Rank 3: metric=0.320287
        {"early_stopping_rounds":10,"learning_rate":0.09544104526717777,"n_estimators":8192,"FLAML_sample_size":49659},  # Rank 4: metric=0.320287
        {"early_stopping_rounds":10,"learning_rate":0.09541180730499482,"n_estimators":8192,"FLAML_sample_size":49659},  # Rank 5: metric=0.320287
    ],

    # Top 5 configurations for extra_tree
    'extra_tree': [
        {"n_estimators":4,"max_features":0.02090605025017727,"max_leaves":4,"criterion":'entropy',"FLAML_sample_size":49659},  # Rank 1: metric=0.233602
        {"n_estimators":5,"max_features":0.02090605025017727,"max_leaves":4,"criterion":'entropy',"FLAML_sample_size":49659},  # Rank 2: metric=0.236307
        {"n_estimators":6,"max_features":0.02090605025017727,"max_leaves":4,"criterion":'entropy',"FLAML_sample_size":49659},  # Rank 3: metric=0.251356
        {"n_estimators":6,"max_features":0.020915738055532117,"max_leaves":4,"criterion":'entropy',"FLAML_sample_size":49659},  # Rank 4: metric=0.251356
        {"n_estimators":4,"max_features":0.02090605025017727,"max_leaves":5,"criterion":'entropy',"FLAML_sample_size":49659},  # Rank 5: metric=0.265905
    ],

    # Top 5 configurations for lgbm
    'lgbm': [
        {"n_estimators":17,"num_leaves":9,"min_child_samples":4,"learning_rate":0.35196178565785524,"log_max_bin":5,"colsample_bytree":1.0,"reg_alpha":0.0010897402303114824,"reg_lambda":0.3461891281239075,"FLAML_sample_size":49659},  # Rank 1: metric=0.248275
        {"n_estimators":19,"num_leaves":4,"min_child_samples":5,"learning_rate":0.8886423277656517,"log_max_bin":7,"colsample_bytree":0.9652473515303718,"reg_alpha":0.001341959440340321,"reg_lambda":0.8835635379011708,"FLAML_sample_size":49659},  # Rank 2: metric=0.249507
        {"n_estimators":4,"num_leaves":4,"min_child_samples":11,"learning_rate":0.6173862843416018,"log_max_bin":7,"colsample_bytree":0.9901327328449183,"reg_alpha":0.002213072229186117,"reg_lambda":0.8635969786167411,"FLAML_sample_size":49659},  # Rank 3: metric=0.258614
        {"n_estimators":4,"num_leaves":4,"min_child_samples":12,"learning_rate":0.6173862843416014,"log_max_bin":7,"colsample_bytree":0.9901327328449183,"reg_alpha":0.002213072229186117,"reg_lambda":0.8635969786167405,"FLAML_sample_size":49659},  # Rank 4: metric=0.258614
        {"n_estimators":14,"num_leaves":4,"min_child_samples":7,"learning_rate":1.0,"log_max_bin":6,"colsample_bytree":1.0,"reg_alpha":0.0032633274946109415,"reg_lambda":4.659344641789399,"FLAML_sample_size":49659},  # Rank 5: metric=0.265003
    ],

    # Top 5 configurations for rf
    'rf': [
        {"n_estimators":18,"max_features":0.02508620447030024,"max_leaves":12,"criterion":'entropy',"FLAML_sample_size":49659},  # Rank 1: metric=0.180855
        {"n_estimators":18,"max_features":0.02513526865143424,"max_leaves":12,"criterion":'entropy',"FLAML_sample_size":49659},  # Rank 2: metric=0.180855
        {"n_estimators":20,"max_features":0.025016171068025294,"max_leaves":12,"criterion":'entropy',"FLAML_sample_size":49659},  # Rank 3: metric=0.186005
        {"n_estimators":20,"max_features":0.025094156728677947,"max_leaves":12,"criterion":'entropy',"FLAML_sample_size":49659},  # Rank 4: metric=0.186005
        {"n_estimators":19,"max_features":0.025342684057687415,"max_leaves":12,"criterion":'entropy',"FLAML_sample_size":49659},  # Rank 5: metric=0.187328
    ],

    # Top 5 configurations for xgb_limitdepth
    'xgb_limitdepth': [
        {"n_estimators":4,"max_depth":7,"min_child_weight":0.6117967126789529,"learning_rate":0.05079105270325453,"subsample":1.0,"colsample_bylevel":0.9125165909431615,"colsample_bytree":0.9823712175288085,"reg_alpha":0.019308192262154213,"reg_lambda":1.1578614055727994,"FLAML_sample_size":49659},  # Rank 1: metric=0.245129
        {"n_estimators":4,"max_depth":5,"min_child_weight":1.7697246184161208,"learning_rate":0.08060208027916571,"subsample":1.0,"colsample_bylevel":0.9576748169038826,"colsample_bytree":0.9411092417205222,"reg_alpha":0.058007843159749536,"reg_lambda":0.35133366158985063,"FLAML_sample_size":10000},  # Rank 2: metric=0.246173
        {"n_estimators":7,"max_depth":6,"min_child_weight":1.738699126940177,"learning_rate":0.09082538366416316,"subsample":0.8385039241494546,"colsample_bylevel":1.0,"colsample_bytree":0.93533553748881,"reg_alpha":0.015604930328895853,"reg_lambda":0.41243772606264223,"FLAML_sample_size":10000},  # Rank 3: metric=0.265100
        {"n_estimators":4,"max_depth":6,"min_child_weight":0.18555676185782113,"learning_rate":0.06117989245544417,"subsample":0.9289575347192514,"colsample_bylevel":1.0,"colsample_bytree":0.9496878891912415,"reg_alpha":0.08524985770352259,"reg_lambda":15.586640991729796,"FLAML_sample_size":49659},  # Rank 4: metric=0.268850
        {"n_estimators":6,"max_depth":7,"min_child_weight":0.36358402068609974,"learning_rate":0.04352117557611909,"subsample":0.9247107965376496,"colsample_bylevel":0.8844932100977313,"colsample_bytree":1.0,"reg_alpha":0.3326145066002043,"reg_lambda":12.695215185331502,"FLAML_sample_size":49659},  # Rank 5: metric=0.305258
    ],

    # Top 5 configurations for xgboost
    'xgboost': [
        {"n_estimators":4,"max_leaves":5,"min_child_weight":2.688726994141466,"learning_rate":0.4376441656114591,"subsample":0.7541193336220411,"colsample_bylevel":0.9939130400196377,"colsample_bytree":0.6597215177184779,"reg_alpha":0.024477447464082414,"reg_lambda":2.885880748906856,"FLAML_sample_size":49659},  # Rank 1: metric=0.118131
        {"n_estimators":4,"max_leaves":5,"min_child_weight":2.949824787067168,"learning_rate":0.4501375974528728,"subsample":0.7706470814398354,"colsample_bylevel":0.9531507952069272,"colsample_bytree":0.6948195213581781,"reg_alpha":0.03739808737099981,"reg_lambda":4.069805750581061,"FLAML_sample_size":49659},  # Rank 2: metric=0.135683
        {"n_estimators":4,"max_leaves":5,"min_child_weight":2.949824787067168,"learning_rate":0.4501375974528721,"subsample":0.7706470814398354,"colsample_bylevel":0.9531507952069272,"colsample_bytree":0.6948195213581781,"reg_alpha":0.03739808737099981,"reg_lambda":4.069805750581055,"FLAML_sample_size":49659},  # Rank 3: metric=0.135683
        {"n_estimators":4,"max_leaves":5,"min_child_weight":1.8249470338508824,"learning_rate":0.37217114693622005,"subsample":0.7454119110994994,"colsample_bylevel":0.9632426545250038,"colsample_bytree":0.7308723461190456,"reg_alpha":0.032098010552263904,"reg_lambda":4.616673840563431,"FLAML_sample_size":49659},  # Rank 4: metric=0.145555
        {"n_estimators":5,"max_leaves":5,"min_child_weight":1.758446990455197,"learning_rate":0.5347621767809498,"subsample":0.7467015597123265,"colsample_bylevel":0.9794829439670584,"colsample_bytree":0.734071300288612,"reg_alpha":0.01696649976631256,"reg_lambda":4.596093086196315,"FLAML_sample_size":49659},  # Rank 5: metric=0.148779
    ],
}
