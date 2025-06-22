/**
 * Theme persistence utilities
 */

export const THEME_STORAGE_KEY = 'tickerml-theme';

export const saveThemePreference = (theme: 'light' | 'dark'): void => {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (error) {
    console.warn('Failed to save theme preference:', error);
  }
};

export const loadThemePreference = (): 'light' | 'dark' => {
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === 'light' || saved === 'dark') {
      return saved;
    }
  } catch (error) {
    console.warn('Failed to load theme preference:', error);
  }
  
  // Default to system preference
  if (typeof window !== 'undefined' && window.matchMedia) {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  
  return 'light';
};

export const getSystemThemePreference = (): 'light' | 'dark' => {
  if (typeof window !== 'undefined' && window.matchMedia) {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  return 'light';
};