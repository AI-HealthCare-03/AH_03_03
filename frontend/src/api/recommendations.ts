import { apiRequest } from "./client";

export type TodayRecommendationItem = {
  title: string;
  description: string;
  reason: string;
  action_type: string;
  related_disease: string;
  priority: number;
};

export type TodayRecommendations = {
  date: string;
  items: TodayRecommendationItem[];
};

export async function getTodayRecommendations(): Promise<TodayRecommendations> {
  return apiRequest<TodayRecommendations>("/recommendations/today");
}
