/**
 * frontend/src/utils/constants.js
 * Color palettes, IMD scale thresholds, and helper utilities.
 */

export const CATEGORY_COLORS = {
  "Low Pressure Area": "#10b981",
  "Monsoon Low": "#06b6d4",
  "Depression": "#22c55e",
  "Deep Depression": "#3b82f6",
  "Cyclonic Storm": "#eab308",
  "Severe Cyclonic Storm": "#f97316",
  "Very Severe Cyclonic Storm": "#ef4444",
  "Extremely Severe Cyclonic Storm": "#dc2626",
  "Super Cyclonic Storm": "#7e22ce",
  "Low": "#22c55e",
  "Moderate": "#3b82f6",
  "Severe": "#f97316",
  "Very Severe": "#ef4444",
  "Extreme": "#7e22ce",
};

export function getCategoryColor(category) {
  if (!category) return "#3b82f6";
  for (const [key, color] of Object.entries(CATEGORY_COLORS)) {
    if (category.toLowerCase().includes(key.toLowerCase())) {
      return color;
    }
  }
  return "#3b82f6";
}

export function formatKnots(kt) {
  return `${Math.round(kt)} kt (${Math.round(kt * 1.852)} km/h)`;
}
