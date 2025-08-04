import { useQuery, type QueryFunctionContext } from "@tanstack/react-query"
import { useMemo } from "react"

import { runMetadataSchema, type RunMetadata } from "@/schema"

export interface RunsQueryParams {
  username?: string | null
  model?: string | null
  runStatus?: "running" | "completed" | "failed" | "all" | null
  sortBy?: "start_time" | "accuracy" | "processing_time" | "total_entries" | null
  sortOrder?: "asc" | "desc" | null
  page: number // 0-based in the hook
  pageSize: number
}

interface RunsResponse {
  data: RunMetadata[]
  total: number
}

const fetchRuns = async ({
  queryKey,
}: QueryFunctionContext<[string, string]>): Promise<RunsResponse> => {
  try {
    const [, p] = queryKey
    const params: RunsQueryParams = JSON.parse(p)

    const qs = new URLSearchParams()
    if (params.username) qs.set("username", params.username)
    if (params.model && params.model !== "all") qs.set("model", params.model)
    if (params.runStatus && params.runStatus !== "all") qs.set("run_status", params.runStatus)
    if (params.sortBy) qs.set("sort_by", params.sortBy)
    if (params.sortOrder) qs.set("sort_order", params.sortOrder)

    qs.set("page", String(params.page + 1)) // API is 1-based
    qs.set("page_size", String(params.pageSize))

    const res = await fetch(`/api/extract/runs?${qs.toString()}`)
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} ${res.statusText}`)
    }

    const data = await res.json()
    const total = Number(res.headers.get("X-Total-Count")) || data.length

    return {
      data: runMetadataSchema.array().parse(data),
      total,
    }
  } catch (error) {
    throw new Error("Failed to fetch runs", { cause: error })
  }
}

export const useRuns = (params: RunsQueryParams) => {
  const queryKey = useMemo<[string, string]>(
    () => ["runs", JSON.stringify(params)],
    [params],
  )

  return useQuery({
    queryKey,
    queryFn: fetchRuns,
  })
}
