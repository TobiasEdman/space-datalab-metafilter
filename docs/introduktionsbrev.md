# Introduktionsbrev

Mall för att presentera metafiltret för partners och kollegor. Ersätt
`[namn]` och anpassa avslutningen. Uppdatera versionsnumret och det som står
under *Vad som faktiskt är verifierat* när en ny release går ut — den
formuleringen är avsiktligt smal och ska inte växa utan att mätningen växer.

---

**Ämne:** Metafilter v0.1.1 – vädermetadata för att välja Sentinel-2-datum

Hej [namn],

Vi har paketerat om metafiltret vi tog fram tillsammans i Space Data Lab 3.0
till ett installerbart Python-bibliotek och släppt en första version. Idén är densamma som tidigare:
använda meteorologiska metadata för att avgöra när det är värt att leta efter
Sentinel-2-bilder, i stället för att hämta allt och sortera bort efteråt.

Installeras med:

```
pip install "git+https://github.com/TobiasEdman/space-datalab-metafilter.git@v0.1.1"
```

Python 3.10 eller senare. Två datakällor stöds: Open-Meteos arkiv kräver inget
konto och räcker gott för att prova, medan CDS-vägen ger full upplösning men
kräver `.cdsapirc`.

Om vad som faktiskt är verifierat: CDS-vägen är körd skarpt mot augusti 2024
över Stockholm och valde då 22 av 31 dagar. Det är en månad och ett område –
inte ett generellt täckningspåstående. Testsviten på 159 tester körs
automatiskt vid varje ändring, och release-noterna skiljer på vad som mätts
live och vad som verifierats offline.

Fullständiga noter och installationsanvisning:
https://github.com/TobiasEdman/space-datalab-metafilter/releases/tag/v0.1.1

Hör gärna av dig om ni provar den, särskilt om ni kör mot andra områden eller
årstider – det är där vi vet minst, och det är precis den sortens återkoppling
som är mest värd.

Hälsningar,
Tobias

---

## Varianter

**Till någon utanför SDL3** — första stycket förutsätter att mottagaren varit
med. Byt det mot en mening som står på egna ben: vad biblioteket gör och att
det kom ur Space Data Lab 3.0.

**Till någon som bara ska veta att det finns** — stryk allt från *"Om vad som
faktiskt är verifierat"* till och med länken, och behåll rubrik, första stycket
och länken.

## Vad brevet medvetet inte gör

Det påstår inget om täckning utöver den månad och det område som faktiskt
mätts. Det kostar en mening och gör resten trovärdig — och du slipper obehaget
om någon kör mot Norrland i februari och får något oväntat.

Det säger heller inget om hur utvecklingen gick till. Granskningsprocessen och
de defekter live-testerna hittade är relevanta internt, inte för mottagaren.
