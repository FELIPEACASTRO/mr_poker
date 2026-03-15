export type ApiJson = Record<string, unknown>;

export type HealthPayload = {
  status: string;
  version: string;
  db_path: string;
};

export type ModuleId = "overview" | "hands" | "sessions" | "benchmarks" | "models";
