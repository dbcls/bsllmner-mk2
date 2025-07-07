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
  useDefaultBsEntries: boolean
  bsEntries?: string | null
  prompt: Prompt[]
  model: string
  maxEntries: number
}

export const defaultFormValues: FormValues = {
  useDefaultBsEntries: true,
  bsEntries: null,
  prompt: [],
  model: "",
  maxEntries: -1,
}
