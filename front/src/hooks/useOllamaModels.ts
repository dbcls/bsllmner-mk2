import { useSuspenseQuery } from "@tanstack/react-query"

import { ollamaModelsSchema, type OllamaModels } from "@/schema"

const fetchOllamaModels = async (): Promise<OllamaModels> => {
  try {
    const res = await fetch(`${BSLLMNER2_OLLAMA_URL}/api/tags`)
    if (!res.ok) {
      throw new Error(`HTTP Error: ${res.status} ${res.statusText}`)
    }
    const data = await res.json()
    return ollamaModelsSchema.parse(data)
  } catch (error) {
    throw new Error("Failed to fetch Ollama models", { cause: error })
  }
}

export const useOllamaModels = () => {
  return useSuspenseQuery<OllamaModels | null, Error>({
    queryKey: ["ollamaModels"],
    queryFn: fetchOllamaModels,
  })
}
