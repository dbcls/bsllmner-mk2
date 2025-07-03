import { useEffect, useState } from "react"

function App() {
  const [content, setContent] = useState<string>("")
  useEffect(() => {
    const fetchContent = async () => {
      const response = await fetch(`${BSLLMNER2_API_URL}/service-info`)
      if (!response.ok) {
        setContent("Failed to fetch content")
        return
      }
      const data = await response.json()
      setContent(JSON.stringify(data) || "No content available")
    }
    fetchContent()
  }, [])

  return (
    <>
      <div>
        {content}
      </div>
    </>
  )
}

export default App
