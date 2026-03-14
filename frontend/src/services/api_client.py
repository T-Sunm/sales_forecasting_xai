"""
API Client for Sales Forecasting Backend
Handles all HTTP communication between Streamlit Frontend and FastAPI Backend
"""

import requests
from typing import Dict, List, Any, Optional
from datetime import date
import streamlit as st

from config import API_BASE_URL


class APIClient:
    """Client to interact with Sales Forecasting Backend API"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or API_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response and errors"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", str(e))
            except:
                error_detail = str(e)
            
            st.error(f"❌ API Error: {error_detail}")
            return {"error": error_detail}
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to backend at {self.base_url}. Is the server running?")
            return {"error": "Connection failed"}
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return {"error": str(e)}
    
    # ==================== HEALTH CHECKS ====================
    
    def health_check(self) -> Dict[str, Any]:
        """Check if backend is alive"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return self._handle_response(response)
        except Exception as e:
            return {"error": str(e), "status": "down"}
    
    def readiness_check(self) -> Dict[str, Any]:
        """Check if backend is ready (models loaded)"""
        try:
            response = self.session.get(f"{self.base_url}/health/ready", timeout=5)
            return self._handle_response(response)
        except Exception as e:
            return {"error": str(e), "status": "not_ready"}
    
    # ==================== DATA ENDPOINTS ====================
    
    def get_stores(self) -> Optional[List[int]]:
        """
        Get list of all available stores
        
        Returns:
            List of store IDs, or None if error
        """
        try:
            response = self.session.get(f"{self.base_url}/prediction/stores", timeout=10)
            data = self._handle_response(response)
            
            if "error" in data:
                return None
            
            return data.get("stores", [])
        except Exception as e:
            st.error(f"Failed to fetch stores: {str(e)}")
            return None
    
    def get_items(self, store_id: int) -> Optional[List[int]]:
        """
        Get list of items available for a specific store
        
        Args:
            store_id: Store identifier
            
        Returns:
            List of item IDs, or None if error
        """
        try:
            response = self.session.get(
                f"{self.base_url}/prediction/items/{store_id}",
                timeout=10
            )
            data = self._handle_response(response)
            
            if "error" in data:
                return None
            
            return data.get("items", [])
        except Exception as e:
            st.error(f"Failed to fetch items for store {store_id}: {str(e)}")
            return None
    
    def get_available_models(self) -> Optional[Dict[str, Any]]:
        """
        Get list of (store_id, item_id) pairs that have trained models
        
        Returns:
            Dict with 'models' list and 'count', or None if error
        """
        try:
            response = self.session.get(
                f"{self.base_url}/models/available",
                timeout=10
            )
            data = self._handle_response(response)
            
            # Check if error occurred
            if not data or data.get("error"):
                return None
            
            return data  # Return full dict with 'models' and 'count'
        except Exception as e:
            st.error(f"Failed to fetch available models: {str(e)}")
            return None
    
    # ==================== PREDICTION ENDPOINT ====================
    
    def predict(
        self,
        store_id: int,
        item_id: int,
        prediction_input: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate sales prediction for a store-item pair
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            prediction_input: Dictionary containing prediction inputs (date, weather, etc.)
            
        Returns:
            Prediction result dict with keys:
                - prediction_value: float
                - store_id: int
                - item_id: int
                - prediction_date: date
                - error: str (if any)
            Or None if request failed
        """
        try:
            # Convert date to string if needed
            if "date" in prediction_input and isinstance(prediction_input["date"], date):
                prediction_input["date"] = prediction_input["date"].isoformat()
            
            # Backend expects flat structure at top level, not nested in "input_data"
            
            # Make POST request
            response = self.session.post(
                f"{self.base_url}/prediction/predict",
                params={
                    "store_id": store_id,
                    "item_id": item_id
                },
                json=prediction_input,  # Send directly
                timeout=300  # 5 minutes - recursive forecasting can take long
            )
            
            result = self._handle_response(response)
            
            if "error" in result:
                return result
            
            return result
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return {"error": str(e)}
    
    # ==================== XAI ENDPOINTS ====================
    
    def get_global_importance(
        self, 
        store_id: int, 
        item_id: int, 
        sample_size: int = 500
    ) -> Optional[Dict]:
        """Get global SHAP feature importance"""
        try:
            response = self.session.post(
                f"{self.base_url}/xai/global_importance",
                params={
                    "store_id": store_id,
                    "item_id": item_id,
                    "sample_size": sample_size
                },
                timeout=60  # SHAP can be slow
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to get global importance: {str(e)}")
            return None

    def get_dependence(
        self, 
        store_id: int, 
        item_id: int, 
        feature: str,
        sample_size: int = 500,
        interaction_feature: Optional[str] = None
    ) -> Optional[Dict]:
        """Get SHAP dependence data for a feature"""
        try:
            params = {
                "store_id": store_id,
                "item_id": item_id,
                "feature": feature,
                "sample_size": sample_size
            }
            if interaction_feature:
                params["interaction_feature"] = interaction_feature
                
            response = self.session.post(
                f"{self.base_url}/xai/dependence",
                params=params,
                timeout=60
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to get dependence data: {str(e)}")
            return None

    def get_local_explanation(
        self, 
        store_id: int, 
        item_id: int, 
        date: Optional[str] = None,
        sample_index: Optional[int] = None,
        top_n: int = 10
    ) -> Optional[Dict]:
        """Get local SHAP explanation for a sample"""
        try:
            body = {"top_n": top_n}
            if date:
                body["date"] = date
            if sample_index is not None:
                body["sample_index"] = sample_index
                
            response = self.session.post(
                f"{self.base_url}/xai/local_explanation",
                params={"store_id": store_id, "item_id": item_id},
                json=body,
                timeout=60
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to get local explanation: {str(e)}")
            return None

    # ==================== UTILITY METHODS ====================
    
    def is_backend_available(self) -> bool:
        """Quick check if backend is reachable"""
        health = self.health_check()
        return health.get("status") == "ok"
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend service information"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            return self._handle_response(response)
        except Exception as e:
            return {"error": str(e)}

    # ==================== DATA SERVICE ENDPOINTS ====================

    def get_historical_data(self, store_id: Optional[int] = None, item_id: Optional[int] = None, limit: int = 1000) -> Dict:
        """Fetch historical data from backend"""
        try:
            params = {"limit": limit}
            if store_id: params["store_id"] = store_id
            if item_id: params["item_id"] = item_id
            
            response = self.session.get(f"{self.base_url}/data/historical", params=params)
            return self._handle_response(response)
        except Exception as e:
            return {"data": [], "error": str(e)}

    def get_top_pair(self) -> Dict:
        """Get the store-item pair with most records from backend"""
        try:
            response = self.session.get(f"{self.base_url}/data/top_pair")
            return self._handle_response(response)
        except Exception as e:
            return {"error": str(e)}

    def get_data_metadata(self) -> Dict:
        """Get dataset metadata from backend"""
        try:
            response = self.session.get(f"{self.base_url}/data/metadata")
            return self._handle_response(response)
        except Exception as e:
            return {"error": str(e)}

    def get_stores_list(self) -> List[int]:
        """Get list of available stores"""
        try:
            response = self.session.get(f"{self.base_url}/data/stores")
            res = self._handle_response(response)
            return res.get("stores", [])
        except Exception:
            return []

    def get_items_list(self, store_id: int) -> List[int]:
        """Get list of items for a specific store"""
        try:
            response = self.session.get(f"{self.base_url}/data/items/{store_id}")
            res = self._handle_response(response)
            return res.get("items", [])
        except Exception:
            return []

    # ==================== ANALYTICS ENDPOINTS ====================

    def get_analytics_filters(self) -> Dict:
        """Get date range and stores for dashboard filters"""
        try:
            response = self.session.get(f"{self.base_url}/analytics/filters")
            return self._handle_response(response)
        except Exception as e:
            return {"error": str(e)}

    def get_analytics_items(self, store_id: Optional[int] = None) -> List[int]:
        """Get items list, optionally filtered by store"""
        try:
            params = {}
            if store_id: params["store_id"] = store_id
            response = self.session.get(f"{self.base_url}/analytics/items", params=params)
            res = self._handle_response(response)
            return res.get("items", [])
        except Exception:
            return []

    def get_kpis_data(self, start_date: str, end_date: str, store_id: Optional[int] = None) -> Dict:
        """Fetch KPI metrics from backend"""
        try:
            params = {"start_date": start_date, "end_date": end_date}
            if store_id: params["store_id"] = store_id
            response = self.session.get(f"{self.base_url}/analytics/kpis", params=params)
            return self._handle_response(response)
        except Exception as e:
            return {"error": str(e)}

    def get_trends_data(self, start_date: str, end_date: str, store_id: Optional[int] = None) -> Dict:
        """Fetch sales trends from backend"""
        try:
            params = {"start_date": start_date, "end_date": end_date}
            if store_id: params["store_id"] = store_id
            response = self.session.get(f"{self.base_url}/analytics/trends", params=params)
            return self._handle_response(response)
        except Exception as e:
            return {"data": [], "error": str(e)}

    def get_performance_data(self, start_date: str, end_date: str, store_id: Optional[int] = None) -> Dict:
        """Fetch performance breakdown (top items/stores)"""
        try:
            params = {"start_date": start_date, "end_date": end_date}
            if store_id: params["store_id"] = store_id
            response = self.session.get(f"{self.base_url}/analytics/performance", params=params)
            return self._handle_response(response)
        except Exception as e:
            return {"top_items": [], "top_stores": [], "error": str(e)}

    def get_comparison_data(self, start_date: str, end_date: str, item_ids: List[int], store_id: Optional[int] = None) -> Dict:
        """Fetch comparison data for multiple products"""
        try:
            params = {
                "start_date": start_date, 
                "end_date": end_date,
                "item_ids": item_ids
            }
            if store_id: params["store_id"] = store_id
            response = self.session.get(f"{self.base_url}/analytics/compare", params=params)
            return self._handle_response(response)
        except Exception as e:
            return {"data": [], "error": str(e)}

    def get_distribution_data(self, start_date: str, end_date: str, store_id: Optional[int] = None, item_id: Optional[int] = None) -> Dict:
        """Fetch sales distribution (detailed & aggregated)"""
        try:
            params = {"start_date": start_date, "end_date": end_date}
            if store_id: params["store_id"] = store_id
            if item_id: params["item_id"] = item_id
            
            response = self.session.get(f"{self.base_url}/analytics/distribution", params=params)
            return self._handle_response(response)
        except Exception as e:
            return {"detailed": [], "aggregated": [], "error": str(e)}



# ==================== SINGLETON INSTANCE ====================

@st.cache_resource
def get_api_client() -> APIClient:
    """
    Get cached API client instance (singleton pattern for Streamlit)
    
    Usage:
        from services import get_api_client
        api = get_api_client()
        stores = api.get_stores()
    """
    return APIClient()
