"""
LLM Explainer Module
Generates natural language insights from SHAP analysis using LLM
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


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
    Generate natural language insights from SHAP analysis using LLM
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "llama-3.3-70b-versatile",
        base_url: str = "https://api.groq.com/openai/v1",
        temperature: float = 0.3
    ):
        """
        Initialize the LLM client
        
        Args:
            api_key: API key for Groq (or set via GROQ_API_KEY env var)
            model: Model name to use
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
        
        # Set environment variable for LangChain
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            temperature=temperature
        )
        
        # System prompt for the analyst persona
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt with feature dictionary"""
        feature_dict_text = "\n".join([
            f"- {k}: {v}" for k, v in FEATURE_DICTIONARY.items()
        ])
        
        return f"""Bạn là một chuyên gia phân tích dữ liệu bán hàng (Sales Analyst) tại Walmart.
Nhiệm vụ của bạn là giải thích kết quả từ mô hình dự báo doanh số cho người quản lý cửa hàng.

Quy tắc:
1. Sử dụng ngôn ngữ tiếng Việt, chuyên nghiệp nhưng dễ hiểu.
2. Tránh thuật ngữ kỹ thuật như "SHAP", "feature importance", "log units".
3. Đưa ra insights có thể hành động được (actionable insights).
4. Viết ngắn gọn, súc tích, dùng bullet points khi cần.
5. Luôn giải thích TẠI SAO chứ không chỉ WHAT.

Từ điển ý nghĩa các biến:
{feature_dict_text}

Các nhóm yếu tố:
- Lịch sử bán hàng: Dựa trên doanh số quá khứ để dự đoán xu hướng.
- Đặc điểm sản phẩm: Hiệu suất của sản phẩm cụ thể.
- Hiệu suất cửa hàng: Doanh số chung của toàn cửa hàng.
- Thời gian & Sự kiện: Ngày trong tuần, tháng, ngày lễ.
- Điều kiện thời tiết: Nhiệt độ, mưa, gió."""

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
        store_nbr: int,
        item_nbr: int,
        importance_df: pd.DataFrame,
        category_summary: pd.DataFrame
    ) -> str:
        """
        Generate natural language report for global feature importance
        
        Args:
            store_nbr: Store ID
            item_nbr: Item ID  
            importance_df: Feature importance DataFrame
            category_summary: Category summary DataFrame
        
        Returns:
            Natural language report string
        """
        # Format data
        data_text = self._format_global_data(importance_df, category_summary)
        
        # Create prompt
        user_prompt = f"""Phân tích mô hình dự báo doanh số cho Cửa hàng {store_nbr}, Sản phẩm {item_nbr}.

Dữ liệu phân tích XAI:
{data_text}

Hãy viết một báo cáo ngắn gọn (3-5 đoạn) bao gồm:
1. Tóm tắt: Mô hình này chủ yếu dựa vào những yếu tố nào?
2. Phân tích: Tại sao các yếu tố này quan trọng? Điều này cho thấy gì về hành vi mua hàng?
3. Khuyến nghị: Người quản lý cửa hàng nên chú ý điều gì?"""

        # Call LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def generate_local_explanation(
        self,
        store_nbr: int,
        item_nbr: int,
        date: str,
        actual_value: float,
        predicted_value: float,
        increasing_factors: pd.DataFrame,
        decreasing_factors: pd.DataFrame
    ) -> str:
        """
        Generate natural language explanation for a specific prediction
        
        Args:
            store_nbr: Store ID
            item_nbr: Item ID
            date: Prediction date
            actual_value: Actual sales
            predicted_value: Predicted sales
            increasing_factors: Factors that increased prediction
            decreasing_factors: Factors that decreased prediction
        
        Returns:
            Natural language explanation string
        """
        # Format data
        data_text = self._format_local_data(
            date, actual_value, predicted_value,
            increasing_factors, decreasing_factors
        )
        
        # Create prompt
        user_prompt = f"""Giải thích dự báo doanh số cho Cửa hàng {store_nbr}, Sản phẩm {item_nbr}.

{data_text}

Hãy giải thích ngắn gọn (2-3 đoạn):
1. Dự báo này cao hay thấp so với thực tế? Tại sao?
2. Yếu tố nào đóng vai trò quyết định? Giải thích theo ngữ cảnh kinh doanh.
3. Có điều gì bất thường cần lưu ý?"""

        # Call LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
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
        
        # Create prompt
        user_prompt = f"""Đánh giá tác động của thời tiết đến doanh số:

{weather_text}

Viết 1 đoạn ngắn (2-3 câu) về vai trò của thời tiết trong dự báo này.
Mô hình có quá phụ thuộc vào thời tiết không? Điều này có hợp lý không?"""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
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