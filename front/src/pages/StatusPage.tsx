import { Box, Typography, Button } from "@mui/material"
import { useRouter } from "@tanstack/react-router"

import CodeBlock from "@/components/CodeBlock"
import Frame from "@/components/Frame"
import OurCard from "@/components/OurCard"
import { getErrorChain } from "@/utils"

interface StatusPageProps {
  type: "error" | "notfound"
  error?: Error
}

export default function StatusPage({ type, error }: StatusPageProps) {
  const router = useRouter()
  const isErrorPage = type === "error"

  const title = isErrorPage
    ? "エラーが発生しました。"
    : "ページが見つかりません"
  const description = isErrorPage
    ? "想定外のエラーが発生しました。お手数ですが、以下の詳細情報を開発者にお伝え下さい。"
    : "お探しのページは存在しないか、移動された可能性があります。\nURL を確認するか、ホームページに戻ってください。"

  let errorMessage = ""
  let errorStack = ""
  if (isErrorPage && error) {
    const chain = getErrorChain(error)
    errorMessage = chain.map((err) => err.message).join("\n\nCaused by: ")
    errorStack = chain.map((err) => err.stack || err.message).join("\n\nCaused by: ")
  }

  const handleRetry = () => {
    if (isErrorPage) {
      router.navigate({ to: router.state.location.pathname, replace: true })
    } else {
      router.navigate({ to: "/" })
    }
  }

  const handleClearAndRetry = () => {
    if (isErrorPage) {
      localStorage.clear()
      router.navigate({ to: router.state.location.pathname, replace: true })
    }
  }

  return (
    <Frame>
      <OurCard sx={{ mt: "1.5rem" }}>
        <Typography sx={{ fontSize: "1.5rem" }} component="h1">
          {title}
        </Typography>
        <Typography sx={{ mt: "0.5rem", whiteSpace: "pre-line" }}>
          {description}
        </Typography>
        {isErrorPage && (
          <>
            <Box sx={{ mt: "1.5rem" }}>
              <Typography sx={{ fontWeight: "bold" }}>
                {"エラーメッセージ"}
              </Typography>
              <CodeBlock
                sx={{ mt: "0.5rem" }}
                content={errorMessage || "No error message available."}
              />
            </Box>
            <Box sx={{ mt: "1rem" }}>
              <Typography sx={{ fontWeight: "bold" }}>
                {"スタックトレース"}
              </Typography>
              <CodeBlock
                sx={{ mt: "0.5rem" }}
                content={errorStack || "No stack trace available."}
              />
            </Box>
          </>
        )}
        <Box sx={{ display: "flex", gap: "1.5rem", mt: "1.5rem" }}>
          <Button variant="contained" color="primary" onClick={handleRetry}>
            {isErrorPage ? "再試行する" : "ホームへ戻る"}
          </Button>
          {isErrorPage && (
            <Button
              variant="contained"
              color="primary"
              onClick={handleClearAndRetry}
              sx={{ textTransform: "none" }}
            >
              {"Cache をクリアして再試行する"}
            </Button>
          )}
        </Box>
      </OurCard>
    </Frame>
  )
}
