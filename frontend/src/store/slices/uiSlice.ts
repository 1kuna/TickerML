import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { DashboardState, Notification } from '../../types';
import { saveThemePreference, loadThemePreference } from '../../utils/theme';

const initialState: DashboardState = {
  sidebarCollapsed: false,
  theme: loadThemePreference(),
  activeMenuItem: 'overview',
  notifications: [],
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    toggleSidebar: (state) => {
      state.sidebarCollapsed = !state.sidebarCollapsed;
    },
    setSidebarCollapsed: (state, action: PayloadAction<boolean>) => {
      state.sidebarCollapsed = action.payload;
    },
    toggleTheme: (state) => {
      state.theme = state.theme === 'light' ? 'dark' : 'light';
      saveThemePreference(state.theme);
    },
    setTheme: (state, action: PayloadAction<'light' | 'dark'>) => {
      state.theme = action.payload;
      saveThemePreference(state.theme);
    },
    setActiveMenuItem: (state, action: PayloadAction<string>) => {
      state.activeMenuItem = action.payload;
    },
    addNotification: (state, action: PayloadAction<Omit<Notification, 'id' | 'timestamp' | 'read'>>) => {
      const notification: Notification = {
        ...action.payload,
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        read: false,
      };
      state.notifications.unshift(notification);
      
      // Keep only last 50 notifications
      if (state.notifications.length > 50) {
        state.notifications = state.notifications.slice(0, 50);
      }
    },
    markNotificationRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload);
      if (notification) {
        notification.read = true;
      }
    },
    markAllNotificationsRead: (state) => {
      state.notifications.forEach(n => n.read = true);
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
  },
});

export const {
  toggleSidebar,
  setSidebarCollapsed,
  toggleTheme,
  setTheme,
  setActiveMenuItem,
  addNotification,
  markNotificationRead,
  markAllNotificationsRead,
  removeNotification,
  clearNotifications,
} = uiSlice.actions;

export default uiSlice.reducer;