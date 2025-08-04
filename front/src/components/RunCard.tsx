import { InputOutlined, RefreshOutlined, InsertChartOutlined } from "@mui/icons-material"
import { Typography, Box, CircularProgress, TableContainer, Table, TableHead, TableRow, TableCell, TableBody, TablePagination, TableSortLabel, TextField, FormControl, Select, MenuItem, InputLabel, Divider, Button, Tabs, Tab, Stack, Dialog, DialogTitle, DialogContent, Checkbox, Paper } from "@mui/material"
import { alpha } from "@mui/material/styles"
import { type SxProps } from "@mui/system"
import { useState } from "react"
import { useFormContext } from "react-hook-form"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
  ScatterChart,
  Scatter,
  LabelList,
} from "recharts"

import AdvancedCodeBlock from "@/components/AdvancedCodeBlock"
import OurCard from "@/components/OurCard"
import { useRunDetail } from "@/hooks/useRunDetail"
import { useRuns } from "@/hooks/useRuns"
import { type FormValues, type MappingValue, type OllamaModels, type RunMetadata } from "@/schema"
import { theme } from "@/theme"

interface RunCardProps {
  sx?: SxProps
  models: OllamaModels
  detailRunName: string | null
  setDetailRunName: (name: string | null) => void
}

interface Filters {
  username: string | null
  model: string | null
  runStatus: "all" | "running" | "completed" | "failed" | null
}

interface MetricPoint {
  timestamp: string
  cpu_percentage: number
  memory_percentage: number
  gpu1_utilization_percentage: number
  gpu1_memory_percentage: number
  gpu1_power_draw: number
  gpu2_utilization_percentage: number
  gpu2_memory_percentage: number
  gpu2_power_draw: number
}

interface PlotData {
  runName: string
  model: string
  thinking: boolean | null
  time: number | null
  timePerEntry: number | null
  totalEntries: number | null
  accuracy: number | null
}

const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: { payload: PlotData }[] }) => {
  if (!active || !payload || payload.length === 0) return null

  const point = payload[0].payload

  return (
    <Paper
      elevation={0}
      sx={{
        p: "1rem",
        border: `1px solid ${theme.palette.divider}`,
        borderRadius: 2,
        backgroundColor: "white",
      }}
    >
      <Stack>
        {[
          ["Run Name", point.runName],
          ["Model", point.model],
          ["Thinking", String(point.thinking)],
          ["Processing Time", point.time != null ? `${point.time}s` : "NA"],
          ["Time per Entry", point.timePerEntry != null ? `${point.timePerEntry.toFixed(2)}s` : "NA"],
          ["Total Entries Num", point.totalEntries ?? "NA"],
          ["Accuracy", point.accuracy != null ? `${point.accuracy.toFixed(2)}%` : "NA"],
        ].map(([label, value]) => (
          <Box key={label} sx={{ display: "flex" }}>
            <Typography sx={{ fontWeight: "bold", minWidth: "10rem" }}>
              {label}:
            </Typography>
            <Typography>{value}</Typography>
          </Box>
        ))}
      </Stack>
    </Paper>
  )
}

function formatTimestamp(ts: string): string {
  if (!/^\d{8}_\d{6}$/.test(ts)) return ts

  const year = parseInt(ts.slice(0, 4), 10)
  const month = parseInt(ts.slice(4, 6), 10) - 1 // JSでは0-indexed
  const day = parseInt(ts.slice(6, 8), 10)
  const hour = parseInt(ts.slice(9, 11), 10)
  const minute = parseInt(ts.slice(11, 13), 10)
  const second = parseInt(ts.slice(13, 15), 10)

  const date = new Date(Date.UTC(year, month, day, hour, minute, second))

  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  })
}

const toMappingTsv = (mapping: Record<string, MappingValue>): string => {
  const headers = ["BioSample ID", "Experiment type", "extraction answer", "mapping answer ID", "mapping answer label"]
  const rows = [
    headers.join("\t"),
    ...Object.entries(mapping).map(([bsId, value]) => {
      return [
        bsId,
        value.experiment_type ?? "",
        value.extraction_answer ?? "",
        value.mapping_answer_id ?? "",
        value.mapping_answer_label ?? "",
      ].join("\t")
    }),
  ]
  return rows.join("\n")
}

export default function RunCard({ sx, models, detailRunName, setDetailRunName }: RunCardProps) {
  const modelNames = models.models.map((model) => model.name)
  const [filters, setFilters] = useState<Filters>({
    username: null, model: "all", runStatus: "all",
  })
  const [filterUsername, setFilterUsername] = useState<string>("")
  const [sortBy, setSortBy] = useState<"start_time" | "accuracy" | "processing_time" | "total_entries" | null>(null)
  const [sortOrder, setSortOrder] = useState<"asc" | "desc" | null>(null)
  const [page, setPage] = useState(0)
  const [rowsPerPage, setRowsPerPage] = useState(10)

  const { data, isLoading, isFetching, refetch } = useRuns({
    ...filters,
    sortBy,
    sortOrder,
    page,
    pageSize: rowsPerPage,
  })

  const handleChangePage = (_event: unknown, newPage: number) => {
    setPage(newPage)
  }

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10))
    setPage(0)
  }

  const createSortHandler = (column: typeof sortBy) => () => {
    if (sortBy === column) {
      setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"))
    } else {
      setSortBy(column)
      setSortOrder("asc")
    }
  }

  const {
    data: detail,
    isLoading: isDetailLoading,
    isFetching: isDetailFetching,
    refetch: refetchDetail,
  } = useRunDetail(detailRunName ?? "")

  const [tab, setTab] = useState(0)
  const [bsEntry, setBsEntry] = useState<string | null>(null)
  const selectedEntry = detail?.input.bs_entries.find((entry) => entry.accession === bsEntry)
  const handleBsEntryOpen = (accession: string) => {
    setBsEntry(accession)
  }
  const handleBsEntryClose = () => {
    setBsEntry(null)
  }
  const [chatResponseBsEntry, setChatResponseBsEntry] = useState<string | null>(null)
  const selectedChatResponse = detail?.output.find((output) => output.accession === chatResponseBsEntry)?.chat_response
  const handleChatResponseOpen = (accession: string) => {
    setChatResponseBsEntry(accession)
  }
  const handleChatResponseClose = () => {
    setChatResponseBsEntry(null)
  }

  const { setValue } = useFormContext<FormValues>()
  const loadToFormInput = () => {
    if (!detail) return
    setValue("useSmallTestData", false)
    setValue("useLargeTestData", false)
    setValue("bsEntries", JSON.stringify(detail.input.bs_entries, null, 2))
    setValue("mapping", toMappingTsv(detail.input.mapping))
    setValue("prompt", detail.input.prompt)
    setValue("model", detail.run_metadata.model)
    setValue("thinking", detail.run_metadata.thinking ?? false)
    setValue("format", detail.input.format ? JSON.stringify(detail.input.format, null, 2) : null)
  }

  const processed: MetricPoint[] = (detail?.metrics ?? []).map((m) => {
    return {
      timestamp: m.timestamp,
      cpu_percentage: m.cpu_percentage,
      memory_percentage: m.memory_used_bytes,
      gpu1_utilization_percentage: m.gpus[0]?.utilization_gpu ?? 0,
      gpu1_memory_percentage: m.gpus[0]?.memory_used_bytes / (m.gpus[0]?.memory_total_bytes ?? 1) * 100,
      gpu1_power_draw: m.gpus[0]?.power_draw ?? 0,
      gpu2_utilization_percentage: m.gpus[1]?.utilization_gpu ?? 0,
      gpu2_memory_percentage: m.gpus[1]?.memory_used_bytes / (m.gpus[1]?.memory_total_bytes ?? 1) * 100,
      gpu2_power_draw: m.gpus[1]?.power_draw ?? 0,
    }
  })
  const charts = [
    { key: "cpu_percentage", label: "CPU Usage (%)" },
    { key: "memory_percentage", label: "Memory Usage (Bytes)" },
    { key: "gpu1_utilization_percentage", label: "GPU1 Utilization (%)" },
    { key: "gpu1_memory_percentage", label: "GPU1 Memory Usage (%)" },
    { key: "gpu1_power_draw", label: "GPU1 Power Draw (W)" },
    { key: "gpu2_utilization_percentage", label: "GPU2 Utilization (%)" },
    { key: "gpu2_memory_percentage", label: "GPU2 Memory Usage (%)" },
    { key: "gpu2_power_draw", label: "GPU2 Power Draw (W)" },
  ]

  // Plot
  const [plottedRuns, setPlottedRuns] = useState<RunMetadata[]>([])
  const togglePlot = (runMetadata: RunMetadata) => {
    setPlottedRuns((prev) => {
      const isPlotted = prev.some((run) => run.run_name === runMetadata.run_name)
      if (isPlotted) {
        return prev.filter((run) => run.run_name !== runMetadata.run_name)
      } else {
        return [...prev, runMetadata]
      }
    })
  }
  const [plotDialogOpen, setPlotDialogOpen] = useState(false)
  const plotData: PlotData[] = plottedRuns.map((run) => ({
    runName: run.run_name,
    model: run.model,
    thinking: run.thinking ?? null,
    time: run.processing_time ?? null,
    totalEntries: run.total_entries ?? null,
    timePerEntry: run.processing_time != null && run.total_entries != null ? (run.processing_time / run.total_entries) : null,
    accuracy: run.accuracy != null ? run.accuracy : 0,
  }))

  if (isLoading) {
    return (
      <OurCard sx={sx}>
        <CircularProgress />
      </OurCard>
    )
  }

  return (
    <OurCard sx={sx} >
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Plot</TableCell>
              <TableCell>Set Detail</TableCell>
              <TableCell>RunName</TableCell>
              <TableCell>Username</TableCell>
              <TableCell>Model</TableCell>
              <TableCell>Thinking</TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortBy === "total_entries"}
                  direction={sortBy === "total_entries" ? sortOrder ?? "asc" : "asc"}
                  onClick={createSortHandler("total_entries")}
                >
                  {"Entries Num"}
                </TableSortLabel>
              </TableCell>
              <TableCell>Completed Num</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortBy === "start_time"}
                  direction={sortBy === "start_time" ? sortOrder ?? "asc" : "asc"}
                  onClick={createSortHandler("start_time")}
                >
                  {"Start Time"}
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortBy === "processing_time"}
                  direction={sortBy === "processing_time" ? sortOrder ?? "asc" : "asc"}
                  onClick={createSortHandler("processing_time")}
                >
                  {"Processing Time (s)"}
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortBy === "accuracy"}
                  direction={sortBy === "accuracy" ? sortOrder ?? "asc" : "asc"}
                  onClick={createSortHandler("accuracy")}
                >
                  {"Accuracy"}
                </TableSortLabel>
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {data?.data.map((run) => (
              <TableRow key={run.run_name}>
                <TableCell>
                  <Checkbox
                    checked={plottedRuns.some((r) => r.run_name === run.run_name)}
                    onChange={() => togglePlot(run)}
                  />
                </TableCell>
                <TableCell>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => {
                      if (run.run_name === "NA") return
                      setDetailRunName(run.run_name)
                    }}
                    disabled={run.run_name === "NA" || detailRunName === run.run_name}
                    sx={{
                      textTransform: "none",
                      minWidth: "7rem",
                    }}
                  >
                    {
                      detailRunName === run.run_name ?
                        "Now Viewing" :
                        "View Details"
                    }
                  </Button>
                </TableCell>
                <TableCell>{run.run_name}</TableCell>
                <TableCell>{run.username ?? "NA"}</TableCell>
                <TableCell>{run.model}</TableCell>
                <TableCell>{String(run.thinking)}</TableCell>
                <TableCell>{run.total_entries ?? "NA"}</TableCell>
                <TableCell>{run.completed_count ?? "NA"}</TableCell>
                <TableCell>{run.status}</TableCell>
                <TableCell>{formatTimestamp(run.start_time)}</TableCell>
                <TableCell>
                  {run.processing_time != null ? run.processing_time.toFixed(1) : "NA"}
                </TableCell>
                <TableCell>
                  {run.accuracy != null ? `${run.accuracy.toFixed(2)}%` : "NA"}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        component="div"
        count={data?.total ?? 0}
        page={page}
        onPageChange={handleChangePage}
        rowsPerPage={rowsPerPage}
        onRowsPerPageChange={handleChangeRowsPerPage}
        rowsPerPageOptions={[10, 25, 50]}
      />

      <Box sx={{ display: "flex", gap: "1.5rem", alignItems: "center", ml: "0.5rem" }}>
        <Typography sx={{ fontWeight: "bold" }}>
          {"Filters:"}
        </Typography>
        <TextField
          label="Username"
          size="small"
          value={filterUsername}
          onChange={(e) => {
            setFilterUsername(e.target.value)
          }}
          onBlur={() => {
            setFilters((prev) => ({ ...prev, username: filterUsername }))
            setPage(0)
          }}
          sx={{ minWidth: "16rem" }}
        />

        <FormControl size="small" sx={{ minWidth: "16rem" }} >
          <InputLabel id="filters-run-status-label">
            {"Run Status"}
          </InputLabel>
          <Select
            id="filters-run-status"
            labelId="filters-run-status-label"
            label="Run Status"
            displayEmpty
            value={filters.runStatus}
            onChange={(e) => {
              setFilters((prev) => ({ ...prev, runStatus: e.target.value as Filters["runStatus"] }))
              setPage(0)
            }}
          >
            {["all", "running", "completed", "failed"].map((v) => (
              <MenuItem key={v} value={v}>
                {v}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: "16rem" }} >
          <InputLabel id="filters-model-label">
            {"Model"}
          </InputLabel>
          <Select
            id="filters-model"
            labelId="filters-model-label"
            label="Model"
            displayEmpty
            value={filters.model}
            onChange={(e) => {
              setFilters((prev) => ({ ...prev, model: e.target.value }))
              setPage(0)
            }}
          >
            {["all", ...modelNames].map((m) => (
              <MenuItem key={m} value={m}>
                {m}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      <Button
        variant="outlined"
        sx={{
          textTransform: "none",
          width: "10rem",
          mt: "1rem",
          ml: "0.5rem",
        }}
        startIcon={<RefreshOutlined />}
        size="small"
        onClick={() => {
          setPage(0)
          refetch()
        }}
        disabled={isFetching}
      >
        {"Reload Table"}
      </Button>
      <Button
        variant="outlined"
        sx={{
          textTransform: "none",
          width: "10rem",
          mt: "1rem",
          ml: "1.5rem",
        }}
        startIcon={<InsertChartOutlined
        />}
        size="small"
        onClick={() => {
          setPlotDialogOpen(true)
        }}
      >
        {"Plot Runs"}
      </Button>

      <Dialog
        open={plotDialogOpen}
        onClose={() => setPlotDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {"Runs Time / Accuracy Plot"}
        </DialogTitle>
        <DialogContent >
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 40 }}>
              <CartesianGrid />
              <XAxis
                type="number"
                dataKey="accuracy"
                name="Accuracy"
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                label={{ value: "Accuracy (%)", position: "bottom", offset: 0 }}
              />
              <YAxis
                type="number"
                dataKey="timePerEntry"
                name="Time per Entry"
                unit="s"
                label={{ value: "Time per Entry (s)", angle: -90, position: "insideLeft" }}
              />
              <Tooltip
                content={<CustomTooltip />}
              />
              <Scatter name="Runs" data={plotData} fill={theme.palette.primary.main}>
                <LabelList dataKey="runName" position="top" />
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </DialogContent>
      </Dialog>

      <Divider sx={{ my: "1.5rem" }} />
      {
        !isDetailLoading && detail && (<>
          <Box sx={{ display: "flex", gap: "1rem", mb: "1rem", ml: "0.5rem" }}>
            <Button
              variant="outlined"
              sx={{
                textTransform: "none",
                width: "10rem",
              }}
              startIcon={<RefreshOutlined />}
              size="small"
              onClick={() => {
                refetchDetail()
              }}
              disabled={isDetailFetching}
            >
              {"Reload Details"}
            </Button>
            <Button
              variant="outlined"
              sx={{
                textTransform: "none",
                width: "10rem",
              }}
              startIcon={<InputOutlined />}
              size="small"
              onClick={loadToFormInput}
            >
              {"Use as Form Input"}
            </Button>
          </Box>

          <Box sx={{ width: "100%" }}>
            <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
              <Tabs
                value={tab}
                onChange={(_, newValue) => setTab(newValue)}
              >
                <Tab label="Metadata" value={0} sx={{ textTransform: "none" }} />
                <Tab label="Input" value={1} sx={{ textTransform: "none" }} />
                <Tab label="Result" value={2} sx={{ textTransform: "none" }} />
                <Tab label="Metrics" value={3} sx={{ textTransform: "none" }} />
                <Tab label="Error" value={4} sx={{ textTransform: "none" }} />
              </Tabs>
            </Box>

            <Box hidden={tab !== 0} sx={{ p: "1rem" }}>
              <Stack spacing={1}>
                {[
                  ["Run Name", detail.run_metadata.run_name],
                  ["Model", detail.run_metadata.model],
                  ["Thinking", String(detail.run_metadata.thinking)],
                  ["Username", detail.run_metadata.username ?? "NA"],
                  ["Start Time", formatTimestamp(detail.run_metadata.start_time)],
                  ["End Time", detail.run_metadata.end_time ? formatTimestamp(detail.run_metadata.end_time) : "NA"],
                  ["Status", detail.run_metadata.status],
                  ["Processing Time", detail.run_metadata.processing_time?.toFixed(1) ?? "NA"],
                  ["Matched Entries Num", detail.run_metadata.matched_entries ?? "NA"],
                  ["Total Entries Num", detail.run_metadata.total_entries ?? "NA"],
                  ["Completed Entries Num", detail.run_metadata.completed_count ?? "NA"],
                  ["Accuracy", detail.run_metadata.accuracy != null ? `${detail.run_metadata.accuracy.toFixed(2)}%` : "NA"],
                ].map(([key, value]) => (
                  <Box key={key} sx={{ display: "flex" }}>
                    <Typography sx={{ fontWeight: "bold", minWidth: "12rem" }}>
                      {key}:
                    </Typography>
                    <Typography>{value}</Typography>
                  </Box>
                ))}
              </Stack>
            </Box>

            <Box hidden={tab !== 1} sx={{ p: "1rem" }}>
              <Typography sx={{ fontWeight: "bold", mb: "0.5rem" }}>
                {"BioSample Entries:"}
              </Typography>
              <AdvancedCodeBlock
                codeString={JSON.stringify(detail.input.bs_entries, null, 2)}
                language="json"
              />
              <Typography sx={{ fontWeight: "bold", mt: "1rem", mb: "0.5rem" }}>
                {"Mapping:"}
              </Typography>
              <AdvancedCodeBlock
                codeString={JSON.stringify(detail.input.mapping, null, 2)}
                language="json"
              />
              <Typography sx={{ fontWeight: "bold", mt: "1rem", mb: "0.5rem" }}>
                {"Prompt:"}
              </Typography>
              <AdvancedCodeBlock
                codeString={JSON.stringify(detail.input.prompt, null, 2)}
                language="json"
              />
              <Typography sx={{ fontWeight: "bold", mt: "1rem", mb: "0.5rem" }}>
                {"Format:"}
              </Typography>
              <AdvancedCodeBlock
                codeString={JSON.stringify(detail.input.format, null, 2)}
                language="json"
              />
            </Box>

            <Box hidden={tab !== 2} sx={{ p: "1rem" }}>
              <>
                <TableContainer component={Box} sx={{
                  border: "1px solid",
                  borderColor: theme.palette.divider,
                  borderRadius: 1,
                }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ fontWeight: "bold" }}>Accession</TableCell>
                        <TableCell sx={{ fontWeight: "bold" }}>Ground Truth (Expected)</TableCell>
                        <TableCell sx={{ fontWeight: "bold" }}>Predicted (Actual)</TableCell>
                        <TableCell sx={{ fontWeight: "bold" }}>Show BS Entry</TableCell>
                        <TableCell sx={{ fontWeight: "bold" }}>Show ChatResponse</TableCell>
                      </TableRow>
                    </TableHead>

                    <TableBody>
                      {detail.evaluation.map((row) => (
                        <TableRow
                          key={row.accession}
                          sx={{
                            bgcolor: row.match ? undefined : alpha(theme.palette.error.main, 0.25),
                          }}
                        >
                          <TableCell>{row.accession}</TableCell>
                          <TableCell>{row.expected ?? "NA"}</TableCell>
                          <TableCell>{row.actual ?? "NA"}</TableCell>
                          <TableCell>
                            <Button
                              variant="outlined"
                              size="small"
                              onClick={() => handleBsEntryOpen(row.accession)}
                              sx={{
                                textTransform: "none",
                                minWidth: "7rem",
                              }}
                            >
                              {"View BS Entry"}
                            </Button>
                          </TableCell>
                          <TableCell>
                            <Button
                              variant="outlined"
                              size="small"
                              onClick={() => handleChatResponseOpen(row.accession)}
                              sx={{
                                textTransform: "none",
                                minWidth: "7rem",
                              }}
                            >
                              {"View ChatResponse"}
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>

                <Dialog
                  open={!!bsEntry}
                  onClose={handleBsEntryClose}
                  maxWidth="md"
                  fullWidth
                >
                  <DialogTitle>Accession: {bsEntry ?? "NA"}</DialogTitle>
                  <DialogContent >
                    <AdvancedCodeBlock
                      codeString={JSON.stringify(selectedEntry, null, 2)}
                      language="json"
                    />
                  </DialogContent>
                </Dialog>

                <Dialog
                  open={!!chatResponseBsEntry}
                  onClose={handleChatResponseClose}
                  maxWidth="md"
                  fullWidth
                >
                  <DialogTitle>ChatResponse: {chatResponseBsEntry ?? "NA"}</DialogTitle>
                  <DialogContent >
                    <AdvancedCodeBlock
                      codeString={JSON.stringify(selectedChatResponse, null, 2)}
                      language="json"
                    />
                  </DialogContent>
                </Dialog>
              </>
            </Box>

            <Box hidden={tab !== 3} sx={{ p: "1rem" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                {charts.map(({ key, label }) => (
                  <Box key={key}>
                    <Typography sx={{ fontWeight: "bold", mb: "0.5rem" }}>
                      {label}
                    </Typography>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={processed}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" hide />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey={key}
                          stroke={theme.palette.primary.main}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                ))}
              </Box>
            </Box>

            <Box hidden={tab !== 4} sx={{ p: "1rem" }}>
              <Typography sx={{ fontWeight: "bold", mb: "0.5rem" }}>
                {"Error Log:"}
              </Typography>
              <AdvancedCodeBlock
                codeString={JSON.stringify(detail.error_log, null, 2)}
                language="json"
              />
            </Box>
          </Box>
        </>)
      }
    </OurCard >
  )
}
