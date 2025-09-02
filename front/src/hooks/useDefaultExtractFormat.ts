import { useQuery } from "@tanstack/react-query"

import { formatSchema, type Format } from "@/schema"

const fetchDefaultExtractFormat = async (): Promise<Format> => {
  try {
    const res = await fetch(`${BSLLMNER2_API_URL}/default-extract-format`)
    if (!res.ok) {
      throw new Error(`HTTP Error: ${res.status} ${res.statusText}`)
    }
    const data = await res.json()
    return formatSchema.parse(data)
  } catch (error) {
    throw new Error("Failed to fetch default extract format", { cause: error })
  }
}

export const useDefaultExtractFormat = () => {
  return useQuery<Format | null, Error>({
    queryKey: ["defaultExtractFormat"],
    queryFn: fetchDefaultExtractFormat,
  })
}
