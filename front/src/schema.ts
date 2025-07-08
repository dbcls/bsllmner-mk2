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

export const PROMPT_ROLES = ["system", "user", "assistant"] as const
export const promptSchema = z.object({
  role: z.enum(PROMPT_ROLES),
  content: z.string(),
})
export type Prompt = z.infer<typeof promptSchema>

export const serviceInfoSchema = z.object({
  api_version: z.string(),
})
export type ServiceInfo = z.infer<typeof serviceInfoSchema>

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
