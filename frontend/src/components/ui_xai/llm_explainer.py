"""
LLM Explainer Module
Generates natural language insights from SHAP analysis using LLM
"""

import os
import base64
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import prompt templates
from .prompts import PromptTemplates, VISUALIZATION_CONTEXTS


# Feature dictionary for translating technical names to business language
FEATURE_DICTIONARY = {
    # Sales History
    "logunits_lag_1": "Doanh số 1 ngày trước",
    "logunits_lag_7": "Doanh số 7 ngày trước",
    "logunits_lag_14": "Doanh số 14 ngày trước",
    "logunits_lag_21": "Doanh số 21 ngày trước",
    "logunits_lag_28": "Doanh số 28 ngày trước",
    "logunits_mean_7d": "Doanh số trung bình 7 ngày",
    "logunits_mean_14d": "Doanh số trung bình 14 ngày",
    "logunits_mean_28d": "Doanh số trung bình 28 ngày",
    "logunits_max_7d": "Doanh số cao nhất 7 ngày",
    "logunits_max_14d": "Doanh số cao nhất 14 ngày",
    "logunits_max_28d": "Doanh số cao nhất 28 ngày",
    "logunits_min_7d": "Doanh số thấp nhất 7 ngày",
    "logunits_min_14d": "Doanh số thấp nhất 14 ngày",
    "logunits_std_7d": "Độ biến động doanh số 7 ngày",
    "logunits_std_14d": "Độ biến động doanh số 14 ngày",
    "logunits_ewma_7d_a05": "Xu hướng doanh số 7 ngày",
    "logunits_ewma_14d_a05": "Xu hướng doanh số 14 ngày",
    
    # Store Context
    "store_sum_7d": "Tổng doanh số cửa hàng 7 ngày",
    "store_mean_7d": "Doanh số trung bình cửa hàng 7 ngày",
    
    # Item Context
    "item_sum_7d": "Tổng doanh số sản phẩm 7 ngày",
    "item_mean_7d": "Doanh số trung bình sản phẩm 7 ngày",
    
    # Calendar
    "day_of_week": "Ngày trong tuần",
    "day": "Ngày trong tháng",
    "month": "Tháng",
    "year": "Năm",
    "is_weekend": "Cuối tuần",
    "is_holiday": "Ngày lễ",
    "is_blackfriday": "Black Friday",
    "season_Spring": "Mùa Xuân",
    "season_Summer": "Mùa Hè",
    "season_Fall": "Mùa Thu",
    "season_Winter": "Mùa Đông",
    
    # Weather
    "tmax": "Nhiệt độ cao nhất",
    "tmin": "Nhiệt độ thấp nhất",
    "preciptotal": "Lượng mưa",
    "stnpressure": "Áp suất khí quyển",
    "sealevel": "Áp suất mực nước biển",
    "resultspeed": "Tốc độ gió",
    "resultdir": "Hướng gió",
    "cool": "Chỉ số làm mát",
}

# Category translations
CATEGORY_TRANSLATIONS = {
    "Sales History": "Lịch sử bán hàng",
    "Store Context": "Hiệu suất cửa hàng",
    "Item Context": "Đặc điểm sản phẩm",
    "Calendar & Events": "Thời gian & Sự kiện",
    "Weather Conditions": "Điều kiện thời tiết",
    "Weather Codes": "Mã thời tiết",
    "Other": "Khác"
}


class SalesInsightGenerator:
    """
    Generate natural language insights from SHAP analysis using LLM.
    Supports both text-only and vision (image+text) inputs.
    """
    
    # Default models
    DEFAULT_TEXT_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    # Image constraints from Groq documentation
    MAX_BASE64_SIZE_MB = 4
    MAX_IMAGES_PER_REQUEST = 5
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        text_model: str = DEFAULT_TEXT_MODEL,
        vision_model: str = DEFAULT_VISION_MODEL,
        base_url: str = "https://api.groq.com/openai/v1",
        temperature: float = 0.3
    ):
        """
        Initialize the Insight Generator with both text and vision capabilities.
        
        Args:
            api_key: API key for Groq (or set via GROQ_API_KEY env var)
            text_model: Model for text-only analysis
            vision_model: Model for image+text analysis
            base_url: API endpoint
            temperature: Creativity level (0-1)
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "API key required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Set environment variable for LangChain compatibility
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Store configuration
        self.text_model = text_model
        self.vision_model = vision_model
        self.base_url = base_url
        self.temperature = temperature
        
        # Lazy initialization for efficiency
        self._text_llm = None
        self._vision_llm = None
        
        # System prompt for analyst persona
        self.system_prompt = self._create_system_prompt()
    
    @property
    def llm(self) -> ChatOpenAI:
        """Get text LLM (for backward compatibility)"""
        return self.text_llm
    
    @property
    def text_llm(self) -> ChatOpenAI:
        """Lazy initialization of text-only LLM"""
        if self._text_llm is None:
            self._text_llm = ChatOpenAI(
                model=self.text_model,
                base_url=self.base_url,
                temperature=self.temperature,
                api_key=self.api_key
            )
        return self._text_llm
    
    @property
    def vision_llm(self) -> ChatOpenAI:
        """Lazy initialization of vision LLM"""
        if self._vision_llm is None:
            self._vision_llm = ChatOpenAI(
                model=self.vision_model,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=1024,
                api_key=self.api_key
            )
        return self._vision_llm
    
    @staticmethod
    def encode_image(image_path: Union[str, Path]) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _create_message_with_image(
        self,
        text: str,
        image_path: Optional[Union[str, Path]] = None
    ) -> HumanMessage:
        """
        Create a HumanMessage with optional image attachment.
        
        Args:
            text: Text content
            image_path: Optional path to image file
        
        Returns:
            HumanMessage with text and/or image
        """
        if image_path is None:
            # Text-only message
            return HumanMessage(content=text)
        
        # Encode image
        base64_image = self.encode_image(image_path)
        
        # Create multimodal message
        return HumanMessage(
            content=[
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        )
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt with feature dictionary"""
        feature_dict_text = "\n".join([
            f"- {k}: {v}" for k, v in FEATURE_DICTIONARY.items()
        ])
        
        return PromptTemplates.SYSTEM_PROMPT.format(
            feature_dict_text=feature_dict_text
        )

    def _translate_feature(self, feature_name: str) -> str:
        """Translate technical feature name to business language"""
        return FEATURE_DICTIONARY.get(feature_name, feature_name)
    
    def _format_global_data(
        self,
        importance_df: pd.DataFrame,
        category_summary: pd.DataFrame,
        top_n: int = 10
    ) -> str:
        """
        Format global importance data for prompt
        
        Args:
            importance_df: DataFrame with feature importance
            category_summary: DataFrame with category-level summary
            top_n: Number of top features to include
        
        Returns:
            Formatted string for prompt
        """
        # Top features
        top_features = importance_df.head(top_n)
        features_text = "Top features quan trọng nhất:\n"
        for i, row in top_features.iterrows():
            translated = self._translate_feature(row['feature'])
            features_text += f"- {translated}: {row['importance']:.4f} (Nhóm: {row['category']})\n"
        
        # Category breakdown
        category_text = "\nTỷ trọng các nhóm yếu tố:\n"
        for _, row in category_summary.iterrows():
            vn_category = CATEGORY_TRANSLATIONS.get(row['category'], row['category'])
            category_text += f"- {vn_category}: {row['importance_pct']:.1f}% ({row['num_features']} biến)\n"
        
        return features_text + category_text
    
    def _format_top_features_data(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20
    ) -> str:
        """
        Format data for Tab 1: Top Features visualization
        
        Args:
            importance_df: Feature importance DataFrame
            top_n: Number of top features (default 20)
        
        Returns:
            Formatted string focusing on feature ranking
        """
        top_features = importance_df.head(top_n)
        
        text = f"TOP {top_n} FEATURES QUAN TRỌNG NHẤT:\n\n"
        for idx, (i, row) in enumerate(top_features.iterrows(), 1):
            translated = self._translate_feature(row['feature'])
            text += f"{idx}. {translated}\n"
            text += f"   - Importance: {row['importance']:.6f}\n"
            text += f"   - Nhóm: {row['category']}\n\n"
        
        # Add gap analysis
        if len(top_features) >= 2:
            top1_importance = top_features.iloc[0]['importance']
            top2_importance = top_features.iloc[1]['importance']
            gap = ((top1_importance - top2_importance) / top1_importance) * 100
            text += f"\nPhân tích: Feature #1 quan trọng hơn #2 khoảng {gap:.1f}%\n"
        
        return text
    
    def _format_categories_data(
        self,
        importance_df: pd.DataFrame,
        category_summary: pd.DataFrame
    ) -> str:
        """
        Format data for Tab 2: Feature Categories visualization
        
        Args:
            importance_df: Feature importance DataFrame
            category_summary: Category summary DataFrame
        
        Returns:
            Formatted string focusing on category breakdown
        """
        text = "PHÂN BỐ THEO NHÓM YẾU TỐ:\n\n"
        
        # Category overview
        for _, row in category_summary.iterrows():
            vn_category = CATEGORY_TRANSLATIONS.get(row['category'], row['category'])
            text += f"📊 {vn_category}:\n"
            text += f"   - Tỷ trọng: {row['importance_pct']:.1f}%\n"
            text += f"   - Số lượng features: {row['num_features']}\n"
            
            # Top features in this category
            cat_features = importance_df[
                importance_df['category'] == row['category']
            ].head(3)
            
            if len(cat_features) > 0:
                text += f"   - Top features:\n"
                for _, feat in cat_features.iterrows():
                    feat_vn = self._translate_feature(feat['feature'])
                    text += f"     • {feat_vn} ({feat['importance']:.4f})\n"
            text += "\n"
        
        return text
    
    def _format_beeswarm_data(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 15
    ) -> str:
        """
        Format data for Tab 3: SHAP Summary (Beeswarm) visualization
        
        Args:
            importance_df: Feature importance DataFrame
            top_n: Number of features shown in plot
        
        Returns:
            Formatted string for beeswarm interpretation guide
        """
        top_features = importance_df.head(top_n)
        
        text = "SHAP SUMMARY - HƯỚNG DẪN ĐỌC BIỂU ĐỒ:\n\n"
        text += "Trong biểu đồ Beeswarm:\n"
        text += "- Mỗi ĐIỂM = 1 mẫu dữ liệu (1 ngày)\n"
        text += "- Trục X = Tác động SHAP (âm: giảm dự báo, dương: tăng dự báo)\n"
        text += "- Màu sắc = Giá trị feature (🔴 đỏ: cao, 🔵 xanh: thấp)\n\n"
        
        text += f"TOP {top_n} FEATURES HIỂN THỊ:\n\n"
        for idx, (i, row) in enumerate(top_features.iterrows(), 1):
            translated = self._translate_feature(row['feature'])
            text += f"{idx}. {translated} (Nhóm: {row['category']})\n"
        
        text += "\nQuan sát các patterns:\n"
        text += "- Phân bố rộng (scatter) = tác động thay đổi nhiều theo context\n"
        text += "- Phân bố hẹp (concentrated) = tác động ổn định\n"
        text += "- Đỏ bên phải + Xanh bên trái = giá trị cao → tăng dự báo (quan hệ dương)\n"
        text += "- Đỏ bên trái + Xanh bên phải = giá trị cao → giảm dự báo (quan hệ âm)\n"
        
        return text
    
    def _format_local_data(
        self,
        date: str,
        actual_value: float,
        predicted_value: float,
        increasing_factors: pd.DataFrame,
        decreasing_factors: pd.DataFrame
    ) -> str:
        """
        Format local explanation data for prompt
        
        Args:
            date: Prediction date
            actual_value: Actual sales value
            predicted_value: Predicted sales value
            increasing_factors: Top factors increasing prediction
            decreasing_factors: Top factors decreasing prediction
        
        Returns:
            Formatted string for prompt
        """
        # Basic info
        error = predicted_value - actual_value
        error_pct = 100 * error / actual_value if actual_value > 0 else 0
        
        text = f"""Ngày dự báo: {date}
        Doanh số thực tế: {actual_value:.1f} sản phẩm
        Doanh số dự báo: {predicted_value:.1f} sản phẩm
        Sai số: {error:+.1f} sản phẩm ({error_pct:+.1f}%)

        """
        
        # Increasing factors
        text += "Các yếu tố TĂNG dự báo:\n"
        for _, row in increasing_factors.iterrows():
            translated = self._translate_feature(row['Feature'])
            text += f"- {translated} = {row['Feature Value']:.2f} → Tác động: +{row['SHAP Impact']:.3f}\n"
        
        # Decreasing factors
        text += "\nCác yếu tố GIẢM dự báo:\n"
        for _, row in decreasing_factors.iterrows():
            translated = self._translate_feature(row['Feature'])
            text += f"- {translated} = {row['Feature Value']:.2f} → Tác động: {row['SHAP Impact']:.3f}\n"
        
        return text
    
    def generate_global_report(
        self,
        store_id: int,
        item_id: int,
        importance_df: pd.DataFrame,
        category_summary: pd.DataFrame,
        image_path: Optional[Union[str, Path]] = None,
        tab_type: str = "categories"
    ) -> str:
        """
        Generate natural language report for global feature importance.
        
        Args:
            store_id: Store ID
            item_id: Item ID  
            importance_df: Feature importance DataFrame
            category_summary: Category summary DataFrame
            image_path: Optional path to visualization image
            tab_type: Type of visualization ("top_features", "categories", "beeswarm")
        
        Returns:
            Natural language report string
        """
        # Format data based on tab type
        if tab_type == "top_features":
            data_text = self._format_top_features_data(importance_df, top_n=20)
            viz_context = VISUALIZATION_CONTEXTS["top_features"]["description"]
            questions = PromptTemplates.GLOBAL_TOP_FEATURES_QUESTIONS
            
        elif tab_type == "categories":
            data_text = self._format_categories_data(importance_df, category_summary)
            viz_context = VISUALIZATION_CONTEXTS["categories"]["description"]
            questions = PromptTemplates.GLOBAL_CATEGORIES_QUESTIONS
            
        else:  # beeswarm
            data_text = self._format_beeswarm_data(importance_df, top_n=15)
            viz_context = VISUALIZATION_CONTEXTS["beeswarm"]["description"]
            questions = PromptTemplates.GLOBAL_BEESWARM_QUESTIONS
        
        # Create base prompt using template
        base_prompt = PromptTemplates.GLOBAL_BASE.format(
            store_nbr=store_id,
            item_nbr=item_id,
            viz_context=viz_context,
            data_text=data_text,
            questions=questions
        )
        
        # Add image context if provided
        if image_path:
            base_prompt += PromptTemplates.IMAGE_CONTEXT_NOTE
        
        # Select appropriate LLM and create message
        if image_path:
            llm = self.vision_llm
            user_message = self._create_message_with_image(base_prompt, image_path)
        else:
            llm = self.text_llm
            user_message = HumanMessage(content=base_prompt)
        
        # Call LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            user_message
        ]
        
        response = llm.invoke(messages)
        return response.content
    
    def generate_local_explanation(
        self,
        store_id: int,
        item_id: int,
        date: str,
        actual_value: float,
        predicted_value: float,
        increasing_factors: pd.DataFrame,
        decreasing_factors: pd.DataFrame,
        image_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate natural language explanation for a specific prediction.
        
        Args:
            store_id: Store ID
            item_id: Item ID
            date: Prediction date
            actual_value: Actual sales
            predicted_value: Predicted sales
            increasing_factors: Factors that increased prediction
            decreasing_factors: Factors that decreased prediction
            image_path: Optional path to waterfall plot image
        
        Returns:
            Natural language explanation string
        """
        # Format data
        data_text = self._format_local_data(
            date, actual_value, predicted_value,
            increasing_factors, decreasing_factors
        )
        
        # Create prompt using template
        base_prompt = PromptTemplates.LOCAL_BASE.format(
            store_nbr=store_id,
            item_nbr=item_id,
            data_text=data_text
        )
        
        # Add image context if provided
        if image_path:
            base_prompt += PromptTemplates.LOCAL_IMAGE_NOTE
        
        # Select appropriate LLM and create message
        if image_path:
            llm = self.vision_llm
            user_message = self._create_message_with_image(base_prompt, image_path)
        else:
            llm = self.text_llm
            user_message = HumanMessage(content=base_prompt)
        
        # Call LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            user_message
        ]
        
        response = llm.invoke(messages)
        return response.content
    
    def generate_weather_impact_summary(
        self,
        importance_df: pd.DataFrame
    ) -> str:
        """
        Generate summary specifically about weather impact
        
        Args:
            importance_df: Feature importance DataFrame
        
        Returns:
            Weather impact summary string
        """
        # Filter weather features
        weather_features = importance_df[
            importance_df['category'].isin(['Weather Conditions', 'Weather Codes'])
        ]
        
        if len(weather_features) == 0:
            return "Thời tiết không có ảnh hưởng đáng kể đến dự báo."
        
        # Format
        weather_text = "Các yếu tố thời tiết:\n"
        for _, row in weather_features.head(5).iterrows():
            translated = self._translate_feature(row['feature'])
            weather_text += f"- {translated}: {row['importance']:.4f}\n"
        
        # Create prompt using template
        base_prompt = PromptTemplates.WEATHER_BASE.format(
            weather_text=weather_text
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=base_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def generate_feature_dependence_analysis(
        self,
        feature_name: str,
        feature_stats: pd.Series,
        category: str,
        store_id: Optional[int] = None,
        item_id: Optional[int] = None,
        image_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate analysis for feature dependence plot.
        
        Args:
            feature_name: Technical name of the feature
            feature_stats: Statistical summary (mean, std, min, max)
            category: Feature category
            store_id: Store ID (optional)
            item_id: Item ID (optional)
            image_path: Optional path to dependence plot image
        
        Returns:
            Natural language explanation of feature impact
        """
        # Translate feature name
        feature_vn = self._translate_feature(feature_name)
        category_vn = CATEGORY_TRANSLATIONS.get(category, category)
        
        # Format statistics
        stats_text = f"""
        YẾU TỐ: {feature_vn} (Danh mục: {category_vn})

        THỐNG KÊ:
        - Giá trị trung bình: {feature_stats['mean']:.2f}
        - Độ lệch chuẩn: {feature_stats['std']:.2f}
        - Giá trị nhỏ nhất: {feature_stats['min']:.2f}
        - Giá trị lớn nhất: {feature_stats['max']:.2f}
        """
        
        # Create context about store/item if provided
        context = ""
        if store_id and item_id:
            context = f"Phân tích cho Cửa hàng {store_id}, Sản phẩm {item_id}.\n\n"
        
        # Create prompt using template
        base_prompt = PromptTemplates.DEPENDENCE_BASE.format(
            context=context,
            stats_text=stats_text
        )
        
        # Add image context if provided
        if image_path:
            base_prompt += PromptTemplates.DEPENDENCE_IMAGE_NOTE
        
        # Select appropriate LLM and create message
        if image_path:
            llm = self.vision_llm
            user_message = self._create_message_with_image(base_prompt, image_path)
        else:
            llm = self.text_llm
            user_message = HumanMessage(content=base_prompt)
        
        # Call LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            user_message
        ]
        
        response = llm.invoke(messages)
        return response.content



# Convenience function for quick use
def create_insight_generator(api_key: str = None) -> Optional[SalesInsightGenerator]:
    """
    Factory function to create SalesInsightGenerator with error handling
    
    Args:
        api_key: Groq API key (optional if set in environment)
    
    Returns:
        SalesInsightGenerator instance or None if failed
    """
    try:
        return SalesInsightGenerator(api_key=api_key)
    except ValueError as e:
        print(f"Warning: Could not initialize LLM explainer: {e}")
        return None