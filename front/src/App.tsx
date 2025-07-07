import CssBaseline from "@mui/material/CssBaseline"
import { ThemeProvider } from "@mui/material/styles"
import { QueryClientProvider, QueryClient } from "@tanstack/react-query"
import { createRootRoute, Router, RouterProvider, createRoute } from "@tanstack/react-router"

import Home from "@/pages/Home"
import StatusPage from "@/pages/StatusPage"
import theme from "@/theme"

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex: number) =>
        Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: Infinity,
      refetchOnMount: "always",
      throwOnError: true,
    },
  },
})

const rootRoute = createRootRoute({
  errorComponent: ({ error }) => (
    <StatusPage type="error" error={error} />
  ),
  notFoundComponent: () => (
    <StatusPage type="notfound" />
  ),
})

const router = new Router({
  routeTree: rootRoute.addChildren([
    createRoute({
      path: "/",
      getParentRoute: () => rootRoute,
      component: Home,
    }),
  ]),
})

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </ThemeProvider>
  )
}
