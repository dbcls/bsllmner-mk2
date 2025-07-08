import { z } from "zod"

export const ollamaModelsSchema = z.object({
  models: z.array(
    z.object({
      name: z.string(),
      model: z.string(),
      modified_at: z.string(),
      size: z.number(),
      digest: z.string(),
      details: z.object({
        parent_model: z.string(),
        format: z.string(),
        family: z.string(),
        families: z.array(z.string()).nullable(),
        parameter_size: z.string(),
        quantization_level: z.string(),
      }),
    }),
  ),
})
export type OllamaModels = z.infer<typeof ollamaModelsSchema>

export const serviceInfoSchema = z.object({
  api_version: z.string(),
})
export type ServiceInfo = z.infer<typeof serviceInfoSchema>

export const PROMPT_ROLES = ["system", "user", "assistant"] as const
export const promptSchema = z.object({
  role: z.enum(PROMPT_ROLES),
  content: z.string(),
})
export type Prompt = z.infer<typeof promptSchema>

export const mappingValueSchema = z.object({
  experiment_type: z.string(),
  extraction_answer: z.string().nullable().optional(),
  mapping_answer_id: z.string().nullable().optional(),
  mapping_answer_label: z.string().nullable().optional(),
})
export type MappingValue = z.infer<typeof mappingValueSchema>

export const wfInputSchema = z.object({
  bs_entries: z.array(z.record(z.any())),
  mapping: z.record(mappingValueSchema),
  prompt: z.array(promptSchema),
  model: z.string(),
  config: z.any(),
  cli_args: z.any().nullable().optional(),
})
export type WorkflowInput = z.infer<typeof wfInputSchema>

export const llmOutputSchema = z.object({
  accession: z.string(),
  output: z.any().nullable().optional(),
  output_full: z.string().nullable().optional(),
  characteristics: z.record(z.any()).nullable().optional(),
  taxId: z.any().nullable().optional(),
  chat_response: z.any(),
})
export type LlmOutput = z.infer<typeof llmOutputSchema>

export const evaluationSchema = z.object({
  accession: z.string(),
  expected: z.string().nullable().optional(),
  actual: z.string().nullable().optional(),
  match: z.boolean(),
})
export type Evaluation = z.infer<typeof evaluationSchema>

export const RUN_STATUS = ["running", "completed", "failed"] as const
export const runMetadataSchema = z.object({
  run_name: z.string(),
  username: z.string().nullable().optional(),
  model: z.string(),
  start_time: z.string(),
  end_time: z.string().nullable().optional(),
  status: z.enum(RUN_STATUS),
  processing_time: z.number().nullable().optional(),
  matched_entries: z.number().nullable().optional(),
  total_entries: z.number().nullable().optional(),
  accuracy: z.number().nullable().optional(),
})
export type RunMetadata = z.infer<typeof runMetadataSchema>

export const errorInfoSchema = z.object({
  type: z.string(),
  message: z.string(),
  traceback: z.string(),
})
export type ErrorInfo = z.infer<typeof errorInfoSchema>

export const errorLogSchema = z.object({
  timestamp: z.string(),
  error: errorInfoSchema,
})
export type ErrorLog = z.infer<typeof errorLogSchema>

export const nvidiaSmiResponseSchema = z.object({
  uuid: z.string(),
  name: z.string(),
  memory_used_bytes: z.number(),
  memory_total_bytes: z.number(),
  utilization_gpu: z.number(),
  power_draw: z.number(),
})
export type NvidiaSmiResponse = z.infer<typeof nvidiaSmiResponseSchema>

export const metricsSchema = z.object({
  timestamp: z.string(),
  block_io_read_bytes: z.number(),
  block_io_write_bytes: z.number(),
  cpu_percentage: z.number(),
  container_name: z.string(),
  container_id: z.string(),
  memory_percentage: z.number(),
  memory_used_bytes: z.number(),
  memory_total_bytes: z.number(),
  net_io_received_bytes: z.number(),
  net_io_sent_bytes: z.number(),
  pids: z.number(),
  gpus: z.array(nvidiaSmiResponseSchema),
})
export type Metrics = z.infer<typeof metricsSchema>

export const resultSchema = z.object({
  input: wfInputSchema,
  output: z.array(llmOutputSchema),
  evaluation: z.array(evaluationSchema),
  metrics: z.array(metricsSchema).nullable().optional(),
  run_metadata: runMetadataSchema,
  error_log: errorLogSchema.nullable().optional(),
})
export type Result = z.infer<typeof resultSchema>

// The form values for the job submission

export interface FormValues {
  useSmallTestData: boolean
  useLargeTestData: boolean
  bsEntries?: string | null
  mapping?: string | null
  prompt: Prompt[]
  model: string
  maxEntries: number
  username: string
  runName: string
}

export const defaultFormValues = (username: string): FormValues => {
  return {
    useSmallTestData: true,
    useLargeTestData: false,
    bsEntries: null,
    mapping: null,
    prompt: [],
    model: "",
    maxEntries: -1,
    username,
    runName: "",
  }
}
