import { AddOutlined, ArrowDownwardOutlined, ArrowUpwardOutlined, DeleteOutlineOutlined, DownloadOutlined, RocketLaunchOutlined } from "@mui/icons-material"
import {
  FormControlLabel, Typography, TextField, Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Button,
  RadioGroup,
  Radio,
  type SelectChangeEvent,
  Divider,
  Snackbar,
  Alert,
} from "@mui/material"
import { type SxProps } from "@mui/system"
import { useQueryClient } from "@tanstack/react-query"
import React, { useState } from "react"
import { useController, useFieldArray, useForm, useFormContext } from "react-hook-form"

import CodeBlock from "@/components/CodeBlock"
import OurCard from "@/components/OurCard"
import OurLink from "@/components/OurLink"
import { usePullOllamaModel } from "@/hooks/usePullOllamaModel"
import { type FormValues, type OllamaModels, type Prompt, PROMPT_ROLES } from "@/schema"
import { getErrorChain } from "@/utils"

interface FormCardProps {
  sx?: SxProps
  models: OllamaModels
  nowStr: string
  setDetailRunName: (name: string | null) => void
  runNames: string[]
}

export default function FormCard({ sx, models, nowStr, setDetailRunName, runNames }: FormCardProps) {
  const { control, setValue, watch, handleSubmit, formState: { isValid } } = useFormContext<FormValues>()

  // === BioSample Entries Form ===
  const useSmallTestData = watch("useSmallTestData")
  const useLargeTestData = watch("useLargeTestData")
  const currentSource: "small" | "large" | "custom" = useSmallTestData ?
    "small" :
    useLargeTestData ?
      "large" :
      "custom"
  const handleSourceChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value as "small" | "large" | "custom"
    setValue("useSmallTestData", value === "small")
    setValue("useLargeTestData", value === "large")
  }
  const { field: bsEntriesField, fieldState: bsEntriesState } = useController({
    name: "bsEntries",
    control,
    rules: {
      validate: (value) => {
        if (currentSource === "custom" && (!value || value.trim() === "")) {
          return "BioSample entries are required when using custom input."
        }
        return true
      },
    },
  })
  const { field: mappingField, fieldState: mappingState } = useController({
    name: "mapping",
    control,
    rules: {
      validate: (value) => {
        if (currentSource === "custom" && (!value || value.trim() === "")) {
          return "Mapping is required when using custom input."
        }
        return true
      },
    },
  })
  const { field: maxEntriesField, fieldState: maxEntriesFieldState } = useController({
    name: "maxEntries",
    control,
    rules: {
      validate: (value) => {
        if (value === "") return true
        const num = Number(value)
        if (Number.isNaN(num)) return "Max entries must be a number"
        if (!Number.isInteger(num)) return "Max entries must be an integer"
        if (num < -1) return "Max entries cannot be less than -1"
        return true
      },
    },
  })

  // === Model Selection Form ===
  const modelNames = models.models.map((model) => model.name)
  const { field: modelField } = useController({
    name: "model",
    control,
  })
  const handleModelChange = (event: SelectChangeEvent<string>) => {
    const newModel = event.target.value
    modelField.onChange(newModel)
    runNameField.onChange(`${newModel}_${nowStr}`)
  }

  const queryClient = useQueryClient()
  const installModelForm = useForm<{ installModel: string }>({
    defaultValues: { installModel: "" },
    mode: "onBlur",
    reValidateMode: "onBlur",
  })
  const {
    control: installModelControl,
    trigger: installModelTrigger,
    reset: resetInstallModelForm,
  } = installModelForm
  const {
    field: installModelField,
    fieldState: installModelState,
  } = useController({
    name: "installModel",
    control: installModelControl,
    rules: {
      required: "Model name is required",
      validate: (value) =>
        !modelNames.includes(value) || `"${value}" is already installed`,
    },
  })
  const installModel = installModelField.value
  const { mutate, isPending } = usePullOllamaModel()
  const installButtonLabel = isPending ? "Installing..." : "Install Model"
  const installButtonDisabled = isPending || !!installModelState.error
  const [installResponse, setInstallResponse] = useState<string | null>(null)
  const installModelHandler = async () => {
    const isValid = await installModelTrigger("installModel", { shouldFocus: true })
    if (!isValid) return

    mutate(installModel, {
      onSuccess: () => {
        setInstallResponse(`Model "${installModel}" installed successfully.`)
        resetInstallModelForm()
        queryClient.invalidateQueries({ queryKey: ["ollamaModels"] })
      },
      onError: (err) => {
        const chain = getErrorChain(err)
        const errorMessages = chain.map((e) => e.message).join("\nCaused by: ")
        setInstallResponse(`Error installing model "${installModel}": ${errorMessages}`)
      },
    })
  }

  const { field: thinkingField } = useController({
    name: "thinking",
    control,
  })

  // === Prompt Form ===
  const {
    fields: promptFields,
    append: appendPrompt,
    remove: removePrompt,
    move: movePrompt,
  } = useFieldArray({
    control,
    name: "prompt",
  })
  const { field: formatField, fieldState: formatFieldState } = useController({
    name: "format",
    control,
    rules: {
      validate: (value) => {
        if (value) {
          try {
            const obj = JSON.parse(value)
            if (typeof obj !== "object" || Array.isArray(obj)) {
              return "Format must be a valid JSON schema object"
            }
          } catch {
            return "Format must be a valid JSON schema object"
          }
        }
        return true
      },
    },
  })

  // === Other Metadata Form ===
  const { field: usernameField, fieldState: usernameState } = useController({
    name: "username",
    control,
    rules: {
      required: "Username is required",
    },
  })
  const { field: runNameField, fieldState: runNameState } = useController({
    name: "runName",
    control,
    rules: {
      required: "Run name is required",
      validate: (value) => {
        if (value.trim() === "") return "Run name cannot be empty"
        if (runNames.includes(value)) return `Run name "${value}" already exists`
        return true
      },
    },
  })
  const [submitState, setSubmitState] = useState<"idle" | "submitting" | "submitted" | "error">("idle")
  const buttonLabel =
    submitState === "submitting"
      ? "Submitting..."
      : submitState === "submitted"
        ? "Submitted"
        : submitState === "error"
          ? "Error"
          : "Submit"
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  // === Submit Handler ===
  const onSubmit = async (values: FormValues) => {
    setSubmitState("submitting")

    try {
      localStorage.setItem("bsllmner2.username", values.username)

      const fd = new FormData()
      fd.append("use_small_test_data", values.useSmallTestData.toString())
      fd.append("use_large_test_data", values.useLargeTestData.toString())
      if (currentSource === "custom") {
        fd.append("bs_entries", new Blob([values.bsEntries ?? ""], { type: "application/json" }))
        fd.append("mapping", new Blob([values.mapping ?? ""], { type: "text/tab-separated-values" }))
      }
      fd.append("prompt", JSON.stringify(values.prompt))
      fd.append("model", values.model)
      if (values.thinking !== null) {
        fd.append("thinking", values.thinking.toString())
      }
      if (values.format) {
        fd.append("format", values.format)
      }
      if (values.maxEntries !== null || values.maxEntries !== "") {
        fd.append("max_entries", values.maxEntries.toString())
      }
      fd.append("username", values.username)
      fd.append("run_name", values.runName)

      const res = await fetch("/api/extract", {
        method: "POST",
        body: fd,
      })
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const data = await res.json()
      // TODO handle the response data
      console.log("Submission successful:", data)
      setDetailRunName(values.runName)
      queryClient.invalidateQueries({ queryKey: ["runs"] })
      queryClient.invalidateQueries({ queryKey: ["run-names"] })
      setSubmitState("submitted")
      setTimeout(() => {
        setSubmitState("idle")
      }, 3000)
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error)
      setErrorMessage(`Submission failed: ${msg}`)
      setSubmitState("error")
      setTimeout(() => {
        setSubmitState("idle")
      }, 3000)
    }
  }

  return (
    <OurCard sx={sx} >
      <Box>
        <Typography component="h1" sx={{ fontSize: "1.5rem" }}>
          {"BioSample Entries"}
        </Typography>
        <Box sx={{ mx: "1.5rem", mt: "0.5rem" }}>
          <RadioGroup
            value={currentSource}
            onChange={handleSourceChange}
          >
            <FormControlLabel
              value="small"
              control={<Radio />}
              label={<>
                {"Use Small Test Data (10 entries, "}
                <OurLink
                  href={"https://github.com/dbcls/bsllmner-mk2/blob/main/tests/test-data/cell_line_example.biosample.json"}
                  text={"bsEntries.json"}
                />
                {", "}
                <OurLink
                  href={"https://github.com/dbcls/bsllmner-mk2/blob/main/tests/test-data/cell_line_example.mapping.tsv"}
                  text={"mapping.tsv"}
                />
                {")"}
              </>}
            />
            <FormControlLabel
              value="large"
              control={<Radio />}
              label={<>
                {"Use Large Test Data (600 entries, "}
                <OurLink
                  href={"https://github.com/dbcls/bsllmner-mk2/blob/main/tests/zenodo-data/biosample_cellosaurus_mapping_testset.json"}
                  text={"bsEntries.json"}
                />
                {", "}
                <OurLink
                  href={"https://github.com/dbcls/bsllmner-mk2/blob/main/tests/zenodo-data/biosample_cellosaurus_mapping_gold_standard.tsv"}
                  text={"mapping.tsv"}
                />
                {")"}
              </>}
            />
            <FormControlLabel
              value="custom"
              control={<Radio />}
              label={"Input your own Entries"}
            />
          </RadioGroup>
          {currentSource === "custom" && (<>
            <Typography sx={{ fontWeight: "bold", mt: "0.5rem" }}>
              {"BioSample Entries JSON:"}
            </Typography>
            <TextField
              {...bsEntriesField}
              value={bsEntriesField.value ?? ""}
              onChange={(e) => bsEntriesField.onChange(e.target.value)}
              placeholder="Enter BioSample entries in JSON format"
              error={!!bsEntriesState.error}
              helperText={bsEntriesState.error?.message}
              multiline
              minRows={5}
              maxRows={10}
              fullWidth
              sx={{ mt: "0.5rem" }}
              slotProps={{
                input: {
                  sx: { fontFamily: "monospace" },
                },
              }}
            />
            <Typography sx={{ fontWeight: "bold", mt: "1rem" }}>
              {"Mapping TSV:"}
            </Typography>
            <TextField
              {...mappingField}
              value={mappingField.value ?? ""}
              onChange={(e) => mappingField.onChange(e.target.value)}
              placeholder="Enter mapping in TSV format"
              error={!!mappingState.error}
              helperText={mappingState.error?.message}
              multiline
              minRows={5}
              maxRows={10}
              fullWidth
              sx={{ mt: "0.5rem" }}
              slotProps={{
                input: {
                  sx: { fontFamily: "monospace" },
                },
              }}
            />
          </>)}
          <TextField
            {...maxEntriesField}
            label="Max Entries"
            value={maxEntriesField.value}
            onChange={(e) => maxEntriesField.onChange(e.target.value)}
            error={!!maxEntriesFieldState.error}
            helperText={
              maxEntriesFieldState.error?.message ??
              "Specify how many entries to process from the beginning (use -1 for no limit)"
            }
            fullWidth
            sx={{
              mt: "1rem",
              maxWidth: "20rem",
              "& .MuiFormHelperText-root": {
                whiteSpace: "nowrap",
              },
            }}
          />
        </Box>
      </Box>

      <Box sx={{ mt: "1.5rem" }}>
        <Typography component="h1" sx={{ fontSize: "1.5rem" }}>
          {"Model"}
        </Typography>
        <Box sx={{ mx: "1.5rem", mt: "1.5rem" }}>
          <FormControl fullWidth>
            <InputLabel id="model-select-label">
              {"Model"}
            </InputLabel>
            <Select
              labelId="model-select-label"
              id="model-select"
              label="Model"
              {...modelField}
              value={modelField.value}
              onChange={handleModelChange}
              sx={{ maxWidth: "20rem" }}
            >
              {modelNames.map((name) => (
                <MenuItem key={name} value={name}>
                  {name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl fullWidth sx={{ maxWidth: "20rem", mt: "1rem" }}>
            <InputLabel id="thinking-select-label">
              {"LLM Thinking Mode"}
            </InputLabel>
            <Select
              labelId="thinking-select-label"
              id="thinking-select"
              label="LLM Thinking Mode"
              {...thinkingField}
              value={String(thinkingField.value)}
              onChange={(e) => {
                const val = e.target.value
                if (val === "null") {
                  thinkingField.onChange(null)
                } else {
                  thinkingField.onChange(val === "true")
                }
              }}
            >
              <MenuItem value="null">None</MenuItem>
              <MenuItem value="true">True</MenuItem>
              <MenuItem value="false">False</MenuItem>
            </Select>
          </FormControl>

          <Box sx={{ mt: "1rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            <Typography>
              {"To add a new model, please select from the "}
              <OurLink
                href={"https://ollama.com/library"}
                text={"Ollama Library"}
              />
              {" and input the model name here."}
            </Typography>
            <Box sx={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
              <TextField
                {...installModelField}
                error={!!installModelState.error}
                helperText={installModelState.error?.message}
                placeholder="Enter model name to install"
                fullWidth
                sx={{ maxWidth: "20rem" }}
              />
              <Button
                variant="outlined"
                sx={{
                  textTransform: "none",
                  mb: installModelState.error ? "1.4rem" : "0",
                  width: "10rem",
                }}
                onClick={installModelHandler}
                startIcon={<DownloadOutlined />}
                disabled={installButtonDisabled}
              >
                {installButtonLabel}
              </Button>
            </Box>
            <Typography sx={{ fontWeight: "bold" }}>
              {"Install Log:"}
            </Typography>
            <CodeBlock
              content={installResponse ?? "No installation response yet."}
            />
          </Box>
        </Box>
      </Box>

      <Box sx={{ mt: "1.5rem" }}>
        <Typography component="h1" sx={{ fontSize: "1.5rem" }}>
          {"Prompt"}
        </Typography>
        <Box sx={{ mx: "1.5rem", mt: "1rem" }}>
          {promptFields.map((field, index) => (
            <Box
              key={field.id}
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: "0.5rem",
                border: "1px solid #ccc",
                borderRadius: 2,
                p: "1rem",
                mb: "1rem",
              }}
            >
              <Box sx={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                <FormControl fullWidth>
                  <InputLabel id={`prompt-${index}-role-label`}>Role</InputLabel>
                  <Select
                    labelId={`prompt-${index}-role-label`}
                    id={`prompt-${index}-role`}
                    label="Role"
                    {...control.register(`prompt.${index}.role` as const)}
                    defaultValue={field.role}
                    sx={{ maxWidth: "20rem" }}
                  >
                    {PROMPT_ROLES.map((role) => (
                      <MenuItem key={role} value={role}>
                        {role}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <IconButton
                  onClick={() => index > 0 && movePrompt(index, index - 1)}
                  disabled={index === 0}
                  sx={{ width: "36px", height: "36px", alignSelf: "center" }}
                >
                  <ArrowUpwardOutlined />
                </IconButton>
                <IconButton
                  onClick={() => index < promptFields.length - 1 && movePrompt(index, index + 1)}
                  disabled={index === promptFields.length - 1}
                  sx={{ width: "36px", height: "36px", alignSelf: "center" }}
                >
                  <ArrowDownwardOutlined />
                </IconButton>
                <IconButton
                  onClick={() => removePrompt(index)}
                  sx={{ width: "36px", height: "36px", alignSelf: "center", mr: "0.5rem" }}
                >
                  <DeleteOutlineOutlined />
                </IconButton>
              </Box>
              <TextField
                multiline
                minRows={2}
                maxRows={5}
                fullWidth
                placeholder="Enter prompt content"
                {...control.register(`prompt.${index}.content` as const)}
                defaultValue={field.content}
              />
            </Box>
          ))}
          <Typography sx={{ fontWeight: "bold", my: "1rem" }}>
            {"NOTE: The last prompt will be extended by concatenating the JSON content of each BioSample Entry."}
          </Typography>
          <Button
            variant="outlined"
            onClick={() =>
              appendPrompt({ role: "user", content: "" } satisfies Prompt)
            }
            sx={{ textTransform: "none" }}
            startIcon={<AddOutlined />}
          >
            {"Add Prompt"}
          </Button>
        </Box>
        <Divider sx={{ my: "1.5rem" }} />
        <Box sx={{ mx: "1.5rem" }}>
          <Typography sx={{ fontWeight: "bold" }}>
            {"Format:"}
          </Typography>
          <TextField
            {...formatField}
            value={formatField.value ?? ""}
            onChange={(e) => formatField.onChange(e.target.value)}
            placeholder="Please enter a JSON schema object for the output format"
            error={!!formatFieldState.error}
            helperText={formatFieldState.error?.message ??
              "Specify the output format as a JSON schema object. Leave empty for default format."
            }
            multiline
            minRows={5}
            maxRows={10}
            fullWidth
            sx={{ mt: "0.5rem" }}
            slotProps={{
              input: {
                sx: { fontFamily: "monospace" },
              },
            }}
          />
        </Box>
      </Box>

      <Box sx={{ mt: "1.5rem" }}>
        <Typography component="h1" sx={{ fontSize: "1.5rem" }}>
          {"Other Metadata"}
        </Typography>
        <Box sx={{ mx: "1.5rem", mt: "1.5rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
          <TextField
            {...usernameField}
            label="Username"
            value={usernameField.value}
            onChange={(e) => usernameField.onChange(e.target.value)}
            error={!!usernameState.error}
            helperText={
              usernameState.error?.message ??
              "Specify your username for this run - no rules, go wild!"
            }
            fullWidth
            sx={{ maxWidth: "30rem" }}
          />
          <TextField
            {...runNameField}
            label="Run Name"
            value={runNameField.value}
            onChange={(e) => runNameField.onChange(e.target.value)}
            error={!!runNameState.error}
            helperText={
              runNameState.error?.message ??
              "Specify a name for this run."
            }
            fullWidth
            sx={{ maxWidth: "30rem" }}
          />
        </Box>
      </Box>
      <Divider sx={{ my: "1.5rem" }} />
      <Button
        variant="contained"
        sx={{ textTransform: "none", width: "10rem" }}
        color={submitState === "error" ? "error" : "primary"}
        onClick={handleSubmit(onSubmit)}
        startIcon={<RocketLaunchOutlined />}
        disabled={submitState !== "idle" || !isValid}
      >
        {buttonLabel}
      </Button>
      <Snackbar
        open={!!errorMessage}
        autoHideDuration={6000}
        onClose={() => setErrorMessage(null)}
        message={errorMessage ?? ""}
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
      >
        <Alert
          onClose={() => setErrorMessage(null)}
          severity="error"
          sx={{ width: "100%" }}
        >
          {errorMessage ?? "An error occurred."}
        </Alert>
      </Snackbar>
    </OurCard >
  )
}
