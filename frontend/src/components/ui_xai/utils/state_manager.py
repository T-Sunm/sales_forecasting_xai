"""
Session state manager for XAI view
Centralized management of Streamlit session state
"""

import streamlit as st


class XAIStateManager:
    """Manage session state for XAI dashboard"""
    
    # State keys
    TAB3_ANALYSIS = 'tab3_show_analysis'
    LAST_SELECTION = 'last_selection'
    
    @staticmethod
    def init_tab_state(tab_name, default_value=False):
        """
        Initialize state for a specific tab
        
        Args:
            tab_name: Name of the state key
            default_value: Default value if not exists
        """
        if tab_name not in st.session_state:
            st.session_state[tab_name] = default_value
    
    @staticmethod
    def reset_on_selection_change(store_nbr, item_nbr):
        """
        Reset relevant states when store/item selection changes
        
        Args:
            store_nbr: Current store number
            item_nbr: Current item number
        
        Returns:
            bool: True if selection changed, False otherwise
        """
        current_selection = f"{store_nbr}_{item_nbr}"
        
        # Initialize last_selection if not exists
        if XAIStateManager.LAST_SELECTION not in st.session_state:
            st.session_state[XAIStateManager.LAST_SELECTION] = current_selection
            return False
        
        # Check if selection changed
        if st.session_state[XAIStateManager.LAST_SELECTION] != current_selection:
            # Selection changed - reset tab3 analysis state
            st.session_state[XAIStateManager.TAB3_ANALYSIS] = False
            st.session_state[XAIStateManager.LAST_SELECTION] = current_selection
            return True
        
        return False
    
    @staticmethod
    def toggle_analysis(tab_name):
        """
        Toggle analysis view for a tab
        
        Args:
            tab_name: State key to toggle
        """
        if tab_name in st.session_state:
            st.session_state[tab_name] = not st.session_state[tab_name]
        else:
            st.session_state[tab_name] = True
    
    @staticmethod
    def get(key, default=None):
        """
        Get a state value
        
        Args:
            key: State key
            default: Default value if key doesn't exist
        
        Returns:
            State value or default
        """
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key, value):
        """
        Set a state value
        
        Args:
            key: State key
            value: Value to set
        """
        st.session_state[key] = value
    
    @staticmethod
    def reset_all():
        """Reset all XAI-related states"""
        keys_to_reset = [
            XAIStateManager.TAB3_ANALYSIS,
            XAIStateManager.LAST_SELECTION
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
