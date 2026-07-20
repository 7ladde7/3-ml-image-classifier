// import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './app.tsx'
import { TamaguiProvider } from '@tamagui/core'
import { config } from '../tamagui.config'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnReconnect: false,
      refetchOnWindowFocus: false,
      retry: false,
      staleTime: Number.MAX_SAFE_INTEGER,
      gcTime: Number.MAX_SAFE_INTEGER,
    }
  }
})
                 
createRoot(document.getElementById('root')!).render(
  <>
    <QueryClientProvider client={queryClient}>
      <TamaguiProvider config={config} defaultTheme="dark">
        <App />
      </TamaguiProvider>
    </QueryClientProvider>
  </>,
)
