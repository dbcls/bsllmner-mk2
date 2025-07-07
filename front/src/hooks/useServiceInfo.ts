import { useQuery } from "@tanstack/react-query"

import { serviceInfoSchema, type ServiceInfo } from "@/schema"

const fetchServiceInfo = async (): Promise<ServiceInfo> => {
  try {
    const res = await fetch(`${BSLLMNER2_API_URL}/service-info`)
    if (!res.ok) {
      throw new Error(`HTTP Error: ${res.status} ${res.statusText}`)
    }
    const data = await res.json()
    return serviceInfoSchema.parse(data)
  } catch (error) {
    throw new Error("Failed to fetch service info", { cause: error })
  }
}

export const useServiceInfo = () => {
  return useQuery<ServiceInfo | null, Error>({
    queryKey: ["serviceInfo"],
    queryFn: fetchServiceInfo,
  })
}
