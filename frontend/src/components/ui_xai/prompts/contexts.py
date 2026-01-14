"""
Visualization contexts and metadata for different chart types
"""

# Visualization contexts for global analysis tabs
VISUALIZATION_CONTEXTS = {
    "top_features": {
        "description": "biểu đồ cột (bar chart) xếp hạng các features",
        "questions_template": "GLOBAL_TOP_FEATURES_QUESTIONS"
    },
    "categories": {
        "description": "biểu đồ phân bố tầm quan trọng theo nhóm yếu tố",
        "questions_template": "GLOBAL_CATEGORIES_QUESTIONS"
    },
    "beeswarm": {
        "description": "biểu đồ SHAP Summary (Beeswarm) với các điểm màu",
        "questions_template": "GLOBAL_BEESWARM_QUESTIONS"
    }
}


# Data formatting configs
DATA_FORMAT_CONFIG = {
    "top_features": {
        "top_n": 20,
        "include_gap_analysis": True
    },
    "categories": {
        "top_features_per_category": 3
    },
    "beeswarm": {
        "top_n": 15,
        "include_interpretation_guide": True
    }
}
