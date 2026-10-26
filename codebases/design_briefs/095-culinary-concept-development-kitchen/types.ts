export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type MenuProject = RecordMeta & WorkRecord & {
  venueId: string;
  creativeBrief: string;
  ingredientInventoryIds: string[];
  serviceWindow: string;
  chefOwnerId: string;
};
export type MenuRequirement = RecordMeta & WorkRecord & {
  caseId: string;
  desiredExperience: string;
  dietaryConstraints: string[];
  equipmentLimits: string[];
  costTarget: Money;
};
export type DishConcept = RecordMeta & WorkRecord & {
  caseId: string;
  requirementIds: string[];
  concept: string;
  ingredientIds: string[];
  preparationApproach: string;
};
export type MenuComposition = RecordMeta & WorkRecord & {
  caseId: string;
  dishIds: string[];
  courseRoles: string[];
  sharedPreparationIds: string[];
  serviceTradeoff: string;
};
export type KitchenProductionPlan = RecordMeta & WorkRecord & {
  caseId: string;
  menuId: string;
  batchQuantities: Quantity[];
  preparationTaskIds: string[];
  totalCost: Money;
};
export type CulinaryRevision = RecordMeta & WorkRecord & {
  caseId: string;
  dishId: string;
  tastingObservations: string[];
  proposedChange: string;
  affectedPlanIds: string[];
};
export type DraftMenuRequirement = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    desiredExperience?: string;
    dietaryConstraints?: string[];
    equipmentLimits?: string[];
    costTarget?: Money;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type IngredientFact={ingredientId:string;lotId:string;revision:string;allergens:string[];dietaryProperties:string[];supplier:string;verified:boolean;substituteFor:string[];};
export type RecipeLine={ingredientId:string;quantityMinor:string;unit:string;yieldServings:string;wasteBasis:string;};
export type Dish={id:string;revision:string;name:string;intent:string;recipe:RecipeLine[];equipment:string[];prepMinutes:number;serviceMinutes:number;dietaryLabels:string[];};
export type KitchenBrief={id:string;serviceTime:string;covers:number;mandatoryRestrictions:string[];preferences:string[];budgetMinor:string;currency:string;equipmentCapacity:Record<string,number>;};
export type KitchenLedger={brief:KitchenBrief;ingredientFacts:IngredientFact[];dishes:Dish[];exactScaledBatches:ScaledBatch[];sharedComponents:string[];schedule:KitchenSlot[];approvedControls:string[];revision:number;};
export type KitchenSlot={equipment:string;taskId:string;startMinute:number;endMinute:number;capacity:number;};
export type MenuFinding={kind:'ingredient'|'allergen'|'coverage'|'yield'|'cost'|'equipment'|'tasting'|'disclosure';subjectId:string;state:'clear'|'blocked'|'review';evidenceIds:string[];detail:string;};
export type KitchenCheckpoint={ledgerRevision:number;recipeRevisions:string[];ingredientRevisions:string[];invalidatedDishIds:string[];costMinor:string;};
export type KitchenProposal={menu:Dish[];findings:MenuFinding[];scaledBatches:ScaledBatch[];sharedPrep:SharedPrep[];disclosures:string[];checkpoint:KitchenCheckpoint;status:'blocked'|'review'|'ready-for-chef';};
export type BatchReconciliation={expected:ScaledBatch[];supplied:ScaledBatch[];discrepancies:string[];blockers:string[];};
export type ScaledBatch={dishId:string;ingredientId:string;quantityMinor:string;unit:string;};
export type SharedPrep={componentId:string;totalQuantityMinor:string;unit:string;usedByDishIds:string[];equipment:string[];};
