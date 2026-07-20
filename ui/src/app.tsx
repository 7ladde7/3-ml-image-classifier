import { XStack, View, Button, YStack, H3, H2, Input, Text, Spinner, H1, RadioGroup, Label } from 'tamagui'
import { useState } from 'react'
import { get, post, BASE_URL } from './utils'

type TPrediction = {url: string; prediction: number; confidence: number}

export const App = () => {
  const [isTraining, setIsTraining] = useState(false)
  const [isPredicting, setIsPredicting] = useState(false)
  const [isSearching, setIsSearching] = useState(false)

  const [prediction, setPrediction] = useState<TPrediction>()

  const [predictUrl, setPredictUrl] = useState("")

  const [trainedAt, setTrainedAt] = useState(0)

  const [label1, setLabel1] = useState("Класс 1")
  const [label2, setLabel2] = useState("Класс 2")
  const [isEditingLabel1, setIsEditingLabel1] = useState(false)
  const [isEditingLabel2, setIsEditingLabel2] = useState(false)

  const [urls1, setUrls1] = useState<string[]>([])
  const [urls2, setUrls2] = useState<string[]>([])

  const [searchInput, setSearchInput] = useState("")

  const [searchResults, setSearchResults] = useState<string[]>([])

  const [activeLabel, setActiveLabel] = useState("1")

  const isTrainable = label1 && label2 && urls1.length && urls2.length

  const train = async () => {
    if (!isTrainable) return
  
    setIsTraining(true)

    setSearchInput("")
    setSearchResults([])

    try {
      await post("/train", [urls1, urls2])

      setTrainedAt(new Date().getTime())
    } catch {}

    setIsTraining(false)
  }

  const predict = async () => {
    if (!predictUrl) return

    setIsPredicting(true)
  
    try {
      const response = await get<TPrediction>(`/predict?url=${predictUrl}`)

      setPrediction(response)
    } catch {}

    setIsPredicting(false)
  }

  const search = async () => {
    if (!searchInput) return
  
    setIsSearching(true)

    try {
      const response = await get<string[]>(`/search?query=${searchInput}`)

      setSearchResults(response)
    } catch {}

    setIsSearching(false)
  }

  return (
    <YStack m="$4">
      <View items="center" justify="center">
        <H2>Классификатор изображений</H2>
      </View>
      <XStack flex={1} gap="$4">
        {trainedAt || isTraining ? null : (
          <View flex={1}>
            <YStack width="100%" items="stretch" gap="$4" p="$4">
              <XStack gap="$2">
                <View flex={1}>
                  <Input value={searchInput} onChange={(ref) => setSearchInput(ref.currentTarget.value)} 
                  onBlur={search} placeholder="Веб-поиск изображений..." />
                </View>
                <View>
                  <Button disabled={isSearching} onClick={search}>{isSearching ? <Spinner /> : "Искать"}</Button>
                </View>
              </XStack>
              {!trainedAt && searchResults.length ? (
                <>
                  <RadioGroup value={activeLabel} onValueChange={setActiveLabel}>
                    <XStack gap="$5" items="center">
                      <XStack gap="$2.5" items="center">
                        <RadioGroup.Item value="1" id="label-1">
                          <RadioGroup.Indicator />
                        </RadioGroup.Item>
                        <Label htmlFor="label-1">{label1}</Label>
                      </XStack>
                      <XStack gap="$2.5" items="center">
                        <RadioGroup.Item value="2" id="label-2">
                          <RadioGroup.Indicator />
                        </RadioGroup.Item>
                        <Label htmlFor="label-2">{label2}</Label>
                      </XStack>
                    </XStack>
                  </RadioGroup>
                  <XStack flexWrap="wrap" justify="space-evenly" gap="$2" items="center">
                    {searchResults.filter(url => ![...urls1, ...urls2].includes(url)).map((url) => (
                      <img src={url} key={url} height={100} onClick={() => {                        
                        const setUrls = activeLabel === "1" ? setUrls1 : setUrls2
                        setUrls(prev => [...new Set([...prev, url])])
                      }} />
                    ))}
                  </XStack>
                </>
              ) : null}
            </YStack>
          </View>
        )}
        <YStack flex={1} p="$4" gap="$4">
          {trainedAt ? (
            <>
              <XStack items="center" justify="space-between"><H3>Обученная модель</H3><Button onClick={() => window.location.reload()}>Сбросить</Button></XStack>
              {prediction ? (
                <>
                  <img src={prediction.url} />
                  <H1 alignSelf="center">{prediction.prediction === 0 ? label1 : label2}</H1>
                  <Text alignSelf="center">Уверенность: {prediction.confidence}</Text>
                </>
              ) : null}
              <Input placeholder="Введите URL изображения..." onChangeText={setPredictUrl} value={predictUrl} />
              <Button variant="secondary" onClick={predict} theme="green" size="$5" disabled={isPredicting}>
                {isPredicting ? <Spinner /> : "Предсказать класс"}
              </Button>
              <H3>Метрики</H3>
              <img src={`${BASE_URL}/artifacts/loss.png?r=${trainedAt}`} />
              <img src={`${BASE_URL}/artifacts/accuracy.png?r=${trainedAt}`} />
            </>
          ) : isTrainable ? (
            <Button variant="secondary" onClick={train} theme="green" size="$6" disabled={isTraining} iconAfter={isTraining ? <Spinner /> : null}>
              {isTraining ? "Обучение..." : "Обучить"}
            </Button>
          ) : null}
          <XStack flex={1} gap="$4">
              <YStack flex={1}>
                {isEditingLabel1 ? <Input onChangeText={setLabel1} value={label1} onBlur={() => setIsEditingLabel1(false)} /> : <H3 style={{ cursor: "pointer" }} onClick={() => setIsEditingLabel1(true)}>{label1} ✏️ <Text color="$gray7">({urls1.length})</Text></H3>}
                <XStack flexWrap="wrap" justify="space-evenly" gap="$2" items="center">
                  {urls1.map(url => <img src={url} height={50} onClick={() => !trainedAt && !isTraining && setUrls1(prev => prev.filter(prevUrl => prevUrl !== url))} />)}
                </XStack>
              </YStack>
              <YStack flex={1}>
                {isEditingLabel2 ? <Input onChangeText={setLabel2} value={label2} onBlur={() => setIsEditingLabel2(false)} /> : <H3 style={{ cursor: "pointer" }}  onClick={() => setIsEditingLabel2(true)}>{label2} ✏️ <Text color="$gray7">({urls2.length})</Text></H3>}
                <XStack flexWrap="wrap" justify="space-evenly" gap="$2" items="center">
                  {urls2.map(url => <img src={url} height={50} onClick={() => !trainedAt && !isTraining && setUrls2(prev => prev.filter(prevUrl => prevUrl !== url))} />)}
                </XStack>
              </YStack>
          </XStack>
        </YStack>
      </XStack>
    </YStack>
  )
}
