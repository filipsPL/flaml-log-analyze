# Warm start configurations for FLAML
# Representative configurations (hybrid selection: top 20.0% + K-Means + best per cluster)
# Auto-generated from optimization logs

warm_start_configs = {
    # Top 5 configurations for catboost
    'catboost': [
        {"early_stopping_rounds":10,"learning_rate":0.14189952377559728,"n_estimators":8192,"FLAML_sample_size":49659},  # Rank 1: metric=0.315344
        {"early_stopping_rounds":10,"learning_rate":0.09544104526717777,"n_estimators":8192,"FLAML_sample_size":49659},  # Rank 2: metric=0.320287
        {"early_stopping_rounds":12,"learning_rate":0.09541333025917802,"n_estimators":8192,"FLAML_sample_size":49659},  # Rank 3: metric=0.320287
        {"early_stopping_rounds":13,"learning_rate":0.12497721195307433,"n_estimators":8192,"FLAML_sample_size":49659},  # Rank 4: metric=0.321610
        {"early_stopping_rounds":10,"learning_rate":0.18406090283225532,"n_estimators":8192,"FLAML_sample_size":49659},  # Rank 5: metric=0.340278
    ],

    # Top 5 configurations for extra_tree
    'extra_tree': [
        {"n_estimators":4,"max_features":0.02090605025017727,"max_leaves":4,"criterion":'entropy',"FLAML_sample_size":49659},  # Rank 1: metric=0.233602
        {"n_estimators":67,"max_leaves":4,"max_features":0.39364864032878705,"criterion":'entropy',"FLAML_sample_size":10000},  # Rank 2: metric=0.271995
        {"n_estimators":4,"max_features":0.038531416567546266,"max_leaves":7,"criterion":'gini',"FLAML_sample_size":10000},  # Rank 3: metric=0.284819
        {"n_estimators":34,"max_leaves":4,"max_features":0.39782392692654595,"criterion":'gini',"FLAML_sample_size":10000},  # Rank 4: metric=0.291623
        {"n_estimators":4,"max_features":0.0364778524567392,"max_leaves":39,"criterion":'entropy',"FLAML_sample_size":49659},  # Rank 5: metric=0.350793
    ],

    # Top 5 configurations for lgbm
    'lgbm': [
        {"n_estimators":17,"num_leaves":9,"min_child_samples":4,"learning_rate":0.35196178565785524,"log_max_bin":5,"colsample_bytree":1.0,"reg_alpha":0.0010897402303114824,"reg_lambda":0.3461891281239075,"FLAML_sample_size":49659},  # Rank 1: metric=0.248275
        {"n_estimators":5,"num_leaves":4,"min_child_samples":14,"learning_rate":0.19007042959642978,"log_max_bin":8,"colsample_bytree":0.8417608937599835,"reg_alpha":0.0009765625,"reg_lambda":1.27426275397643,"FLAML_sample_size":49659},  # Rank 2: metric=0.287310
        {"n_estimators":17,"num_leaves":4,"min_child_samples":14,"learning_rate":0.059348124744491845,"log_max_bin":8,"colsample_bytree":0.9865279165705738,"reg_alpha":0.01336233483179995,"reg_lambda":0.626157827177311,"FLAML_sample_size":10000},  # Rank 3: metric=0.302118
        {"n_estimators":5,"num_leaves":5,"min_child_samples":6,"learning_rate":0.5648426064926187,"log_max_bin":7,"colsample_bytree":0.8522523296440149,"reg_alpha":0.0041956320927146315,"reg_lambda":10.218186813704232,"FLAML_sample_size":49659},  # Rank 4: metric=0.302333
        {"n_estimators":7,"num_leaves":13,"min_child_samples":4,"learning_rate":0.8012754274158277,"log_max_bin":4,"colsample_bytree":1.0,"reg_alpha":0.0009765625,"reg_lambda":3.9303112088755325,"FLAML_sample_size":49659},  # Rank 5: metric=0.323238
    ],

    # Top 5 configurations for rf
    'rf': [
        {"n_estimators":18,"max_features":0.02508620447030024,"max_leaves":12,"criterion":'entropy',"FLAML_sample_size":49659},  # Rank 1: metric=0.180855
        {"n_estimators":10,"max_leaves":11,"max_features":0.3676318363023739,"criterion":'gini',"FLAML_sample_size":49659},  # Rank 2: metric=0.205944
        {"n_estimators":16,"max_features":0.024657669020399484,"max_leaves":13,"criterion":'gini',"FLAML_sample_size":49659},  # Rank 3: metric=0.239116
        {"n_estimators":20,"max_features":0.024621038717806135,"max_leaves":13,"criterion":'gini',"FLAML_sample_size":49659},  # Rank 4: metric=0.240951
        {"n_estimators":12,"max_features":0.026634922273411587,"max_leaves":8,"criterion":'gini',"FLAML_sample_size":49659},  # Rank 5: metric=0.265055
    ],

    # Top 5 configurations for xgb_limitdepth
    'xgb_limitdepth': [
        {"n_estimators":4,"max_depth":7,"min_child_weight":0.6117967126789529,"learning_rate":0.05079105270325453,"subsample":1.0,"colsample_bylevel":0.9125165909431615,"colsample_bytree":0.9823712175288085,"reg_alpha":0.019308192262154213,"reg_lambda":1.1578614055727994,"FLAML_sample_size":49659},  # Rank 1: metric=0.245129
        {"n_estimators":7,"max_depth":6,"min_child_weight":1.738699126940177,"learning_rate":0.09082538366416316,"subsample":0.8385039241494546,"colsample_bylevel":1.0,"colsample_bytree":0.93533553748881,"reg_alpha":0.015604930328895853,"reg_lambda":0.41243772606264223,"FLAML_sample_size":10000},  # Rank 2: metric=0.265100
        {"n_estimators":4,"max_depth":6,"min_child_weight":0.18555676185782113,"learning_rate":0.06117989245544417,"subsample":0.9289575347192514,"colsample_bylevel":1.0,"colsample_bytree":0.9496878891912415,"reg_alpha":0.08524985770352259,"reg_lambda":15.586640991729796,"FLAML_sample_size":49659},  # Rank 3: metric=0.268850
        {"n_estimators":7,"max_depth":6,"min_child_weight":5.513104238121017,"learning_rate":0.02613368731088021,"subsample":1.0,"colsample_bylevel":0.9418743732834765,"colsample_bytree":0.9624964758521868,"reg_alpha":0.031247113458966316,"reg_lambda":4.99506976101826,"FLAML_sample_size":10000},  # Rank 4: metric=0.306127
        {"n_estimators":14,"max_depth":7,"min_child_weight":0.33878650012704536,"learning_rate":0.07950973215385125,"subsample":1.0,"colsample_bylevel":1.0,"colsample_bytree":0.9018622848259131,"reg_alpha":0.0017151776630213724,"reg_lambda":0.2576278386874353,"FLAML_sample_size":49659},  # Rank 5: metric=0.320053
    ],

    # Top 5 configurations for xgboost
    'xgboost': [
        {"n_estimators":4,"max_leaves":5,"min_child_weight":2.688726994141466,"learning_rate":0.4376441656114591,"subsample":0.7541193336220411,"colsample_bylevel":0.9939130400196377,"colsample_bytree":0.6597215177184779,"reg_alpha":0.024477447464082414,"reg_lambda":2.885880748906856,"FLAML_sample_size":49659},  # Rank 1: metric=0.118131
        {"n_estimators":4,"max_leaves":5,"min_child_weight":1.8249470338508824,"learning_rate":0.37217114693622005,"subsample":0.7454119110994994,"colsample_bylevel":0.9632426545250038,"colsample_bytree":0.7308723461190456,"reg_alpha":0.032098010552263904,"reg_lambda":4.616673840563431,"FLAML_sample_size":49659},  # Rank 2: metric=0.145555
        {"n_estimators":4,"max_leaves":5,"min_child_weight":1.2607744733305413,"learning_rate":0.26905431716482003,"subsample":0.7619935617778097,"colsample_bylevel":1.0,"colsample_bytree":0.7555739125194842,"reg_alpha":0.1386714242535025,"reg_lambda":4.95085596509753,"FLAML_sample_size":49659},  # Rank 3: metric=0.159975
        {"n_estimators":22,"max_leaves":16,"min_child_weight":11.21289347824816,"learning_rate":0.7680442808252305,"subsample":0.6456201552485976,"colsample_bylevel":0.10870367449081778,"colsample_bytree":0.26447350307495754,"reg_alpha":0.023274208719392404,"reg_lambda":24.42201465165715,"FLAML_sample_size":49659},  # Rank 4: metric=0.204167
        {"n_estimators":8,"max_leaves":155,"min_child_weight":4.752669988525024,"learning_rate":0.8212861440075813,"subsample":0.640738810755572,"colsample_bylevel":0.07644748683275493,"colsample_bytree":0.4233203703089161,"reg_alpha":0.013392180877007996,"reg_lambda":118.89923963477102,"FLAML_sample_size":49659},  # Rank 5: metric=0.225228
    ],
}
