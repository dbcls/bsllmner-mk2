import { Box, Container } from "@mui/material"

import AppFooter from "@/components/AppFooter"
import AppHeader from "@/components/AppHeader"

interface FrameProps {
  children: React.ReactNode
}

export default function Frame({ children }: FrameProps) {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <AppHeader />
      <Container component="main" maxWidth="lg" sx={{ flexGrow: 1 }}>
        {children}
      </Container>
      <AppFooter />
    </Box>
  )
}
