import { apiRequest, type ApiValue } from "./client";

export type DietFoodItem = {
  name: string;
  quantity?: string | null;
  memo?: string | null;
};

export type DietNutritionSummary = {
  calories?: number | null;
  carbohydrate_g?: number | null;
  protein_g?: number | null;
  fat_g?: number | null;
  sodium_mg?: number | null;
  disease_scores?: Record<string, number | null>;
  scoring_source?: string | null;
  explanation?: Record<string, ApiValue> | null;
};

export type DietAnalyzeResponse = {
  message: string;
  diet_record: Record<string, ApiValue>;
  photo_result: Record<string, ApiValue>;
  detected_foods: Array<Record<string, ApiValue>>;
  nutrition_summary: DietNutritionSummary;
  diet_score: number;
  diet_feedback: string;
  disease_scores?: Record<string, number | null> | null;
  food_score_details?: Array<Record<string, ApiValue>>;
  scoring_source?: string | null;
  explanation?: Record<string, ApiValue> | null;
  warnings?: string[];
  recommended_actions?: string[];
};

export type DietRecordPayload = {
  meal_type?: string | null;
  meal_time?: string | null;
  description?: string | null;
  image_path?: string | null;
  detected_foods?: DietFoodItem[] | Record<string, ApiValue> | null;
  nutrition_summary?: DietNutritionSummary | null;
  diet_score?: number | null;
  diet_feedback?: string | null;
  analysis_method?: string | null;
  is_user_corrected?: boolean;
  memo?: string | null;
};

export async function listDietRecords<T>(): Promise<T> {
  return apiRequest<T>("/diets");
}

export async function getDietRecord<T>(dietRecordId: number): Promise<T> {
  return apiRequest<T>(`/diets/${dietRecordId}`);
}

export async function listDietPhotoResults<T>(dietRecordId: number): Promise<T> {
  return apiRequest<T>(`/diets/${dietRecordId}/photo-result`);
}

export async function createDietRecord<T>(payload: DietRecordPayload): Promise<T> {
  return apiRequest<T>("/diets", { method: "POST", body: payload as Record<string, ApiValue> });
}

export async function analyzeDiet<T>(payload: Record<string, ApiValue>): Promise<T> {
  return apiRequest<T>("/diets/analyze", { method: "POST", body: payload });
}
