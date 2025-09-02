import { useEffect, useMemo, useState } from "react"
import { FormProvider, useForm } from "react-hook-form"

import FormCard from "@/components/FormCard"
import Frame from "@/components/Frame"
import Loading from "@/components/Loading"
import RunCard from "@/components/RunCard"
import { useDefaultExtractFormat } from "@/hooks/useDefaultExtractFormat"
import { useDefaultExtractPrompt } from "@/hooks/useDefaultExtractPrompt"
import { useOllamaModels } from "@/hooks/useOllamaModels"
import { useRunNames } from "@/hooks/useRunNames"
import { useServiceInfo } from "@/hooks/useServiceInfo"
import { type FormValues, defaultFormValues } from "@/schema"
import { getNowStr } from "@/utils"

const USERNAME_LS_KEY = "bsllmner2.username"

export default function Home() {
  const serviceInfoQuery = useServiceInfo()
  const ollamaModelsQuery = useOllamaModels()
  const defaultExtractPromptQuery = useDefaultExtractPrompt()
  const defaultExtractFormatQuery = useDefaultExtractFormat()
  const runNamesQuery = useRunNames()
  const loading = serviceInfoQuery.isLoading ||
    ollamaModelsQuery.isLoading ||
    defaultExtractPromptQuery.isLoading ||
    defaultExtractFormatQuery.isLoading ||
    runNamesQuery.isLoading

  const storedUsername = useMemo(
    () => localStorage.getItem(USERNAME_LS_KEY) ?? "triceratops",
    [],
  )
  const nowStr = useMemo(
    () => getNowStr(),
    [],
  )
  const [detailRunName, setDetailRunName] = useState<string | null>(null)

  const methods = useForm<FormValues>({
    defaultValues: defaultFormValues(storedUsername),
    mode: "onBlur",
    reValidateMode: "onBlur",
  })

  // Initialize default values based on the fetched data
  useEffect(() => {
    if (defaultExtractPromptQuery.data) {
      methods.setValue("prompt", defaultExtractPromptQuery.data, { shouldValidate: true })
    }
  }, [methods, defaultExtractPromptQuery.data])
  useEffect(() => {
    if (defaultExtractFormatQuery.data) {
      methods.setValue("format", JSON.stringify(defaultExtractFormatQuery.data, null, 2), { shouldValidate: true })
    }
  }, [methods, defaultExtractFormatQuery.data])
  useEffect(() => {
    if (ollamaModelsQuery.data) {
      const modelName = ollamaModelsQuery.data.models[0].name
      const runName = `${modelName}_${nowStr}`
      methods.setValue("model", modelName, { shouldValidate: true })
      methods.setValue("runName", runName, { shouldValidate: true })
    }
  }, [methods, ollamaModelsQuery.data, nowStr])

  if (loading) {
    return <Frame><Loading msg="Loading..." /></Frame>
  }

  return (
    <Frame>
      <FormProvider {...methods}>
        <FormCard sx={{ my: "1.5rem" }} models={ollamaModelsQuery.data!} nowStr={nowStr} setDetailRunName={setDetailRunName} runNames={runNamesQuery.data!} />
        <RunCard sx={{ my: "1.5rem" }} models={ollamaModelsQuery.data!} detailRunName={detailRunName} setDetailRunName={setDetailRunName} />
      </FormProvider>
    </Frame>
  )
}
