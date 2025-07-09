import { useQuery } from "@tanstack/react-query"
import { z } from "zod"

const fetchRunNames = async (): Promise<string[]> => {
  const res = await fetch("/api/extract/run-names")
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${res.statusText}`)
  }

  const data = await res.json()
  return z.array(z.string()).parse(data)
}

export const useRunNames = () => {
  return useQuery({
    queryKey: ["run-names"],
    queryFn: fetchRunNames,
    retry: 1,
  })
}
