import { useQuery } from "@tanstack/react-query"

import { promptSchema, type Prompt } from "@/schema"

const fetchDefaultExtractPrompt = async (): Promise<Prompt[]> => {
  try {
    const res = await fetch(`${BSLLMNER2_API_URL}/default-extract-prompt`)
    if (!res.ok) {
      throw new Error(`HTTP Error: ${res.status} ${res.statusText}`)
    }
    const data = await res.json()
    return data.map((item: unknown) => promptSchema.parse(item))
  } catch (error) {
    throw new Error("Failed to fetch default extract prompt", { cause: error })
  }
}

export const useDefaultExtractPrompt = () => {
  return useQuery<Prompt[] | null, Error>({
    queryKey: ["defaultExtractPrompt"],
    queryFn: fetchDefaultExtractPrompt,
  })
}
