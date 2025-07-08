import { useQuery, type QueryFunctionContext } from "@tanstack/react-query"

import { resultSchema, type Result } from "@/schema"

const fetchRunDetail = async ({
  queryKey,
}: QueryFunctionContext<[string, string]>): Promise<Result> => {
  const [, runName] = queryKey

  const res = await fetch(`/api/extract/runs/${encodeURIComponent(runName)}`)
  if (!res.ok) {
    if (res.status === 404) {
      throw new Error(`Run "${runName}" not found`)
    }
    throw new Error(`HTTP ${res.status} ${res.statusText}`)
  }

  const json = await res.json()
  return resultSchema.parse(json)
}

export const useRunDetail = (runName: string) => {
  return useQuery({
    queryKey: ["run", runName],
    queryFn: fetchRunDetail,
    enabled: !!runName,
    retry: 1,
  })
}
