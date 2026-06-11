import { apiBlobRequest, apiRequest, type ApiValue } from "./client";
import type { AsyncJob } from "./jobs";

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

export type DietCandidateNutrition = {
  energy_kcal?: number | null;
  carbohydrate_g?: number | null;
  protein_g?: number | null;
  fat_g?: number | null;
  sodium_mg?: number | null;
};

export type DietNutritionCandidate = {
  candidate_id?: string | null;
  food_code?: string | null;
  food_name?: string | null;
  source?: string | null;
  match_status?: string | null;
  serving_size?: string | null;
  nutrition_preview?: DietCandidateNutrition | null;
};

export type DietCandidateFood = {
  food_item_id?: string | null;
  vision_food_name?: string | null;
  raw_food_name?: string | null;
  display_name?: string | null;
  food_code?: string | null;
  source?: string | null;
  nutrition_status?: string | null;
  match_status?: string | null;
  auto_confirmed?: boolean | null;
  needs_user_confirmation?: boolean | null;
  selected_candidate?: DietNutritionCandidate | null;
  candidates?: DietNutritionCandidate[];
  nutrition?: DietCandidateNutrition | null;
  editable?: boolean | null;
  user_action?: string | null;
  message?: string | null;
};

export type DietCandidateSummary = {
  detected_food_count?: number | null;
  auto_confirmed_count?: number | null;
  needs_user_confirmation_count?: number | null;
  no_candidates_count?: number | null;
  total_energy_kcal?: number | null;
  total_carbohydrate_g?: number | null;
  total_protein_g?: number | null;
  total_fat_g?: number | null;
  total_sodium_mg?: number | null;
  nutrition_calculation_status?: string | null;
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
  vision_provider?: string | null;
  fallback_used?: boolean;
  raw_output?: Record<string, ApiValue> | null;
  explanation?: Record<string, ApiValue> | null;
  warnings?: string[];
  recommended_actions?: string[];
  summary?: DietCandidateSummary | null;
  auto_confirmed_foods?: DietCandidateFood[];
  needs_confirmation_foods?: DietCandidateFood[];
  no_candidate_foods?: DietCandidateFood[];
  nutrition_calculation_status?: string | null;
  needs_user_confirmation?: boolean | null;
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

export async function getDietRecordImage(dietRecordId: number): Promise<Blob> {
  return apiBlobRequest(`/diets/${dietRecordId}/image`);
}

export async function listDietPhotoResults<T>(dietRecordId: number): Promise<T> {
  return apiRequest<T>(`/diets/${dietRecordId}/photo-result`);
}

export async function createDietRecord<T>(payload: DietRecordPayload): Promise<T> {
  return apiRequest<T>("/diets", { method: "POST", body: payload as Record<string, ApiValue> });
}

export async function analyzeDiet(payload: Record<string, ApiValue> | FormData): Promise<AsyncJob> {
  return apiRequest<AsyncJob>("/diets/analyze", { method: "POST", body: payload });
}
