# Instrucciones para benchmarks comparativos

Comparamos tres métodos de sanitización: SanText (baseline), Ours (Normal) y Ours++ (Plus).
La idea es variar un parámetro a la vez, fijando el resto en valores por defecto.

## Parámetros por defecto

| Parámetro | Valor | Nota |
|-----------|-------|------|
| dataset | i2b2 | `--task i2b2 --language en` |
| seed | 42 | Un solo seed para exploración; 5 seeds (1, 21, 42, 84, 132) para resultados finales |
| s_epsilon | epsilon/2 | Siempre la mitad del epsilon |
| p | 0.7 | Solo aplica a Plus |
| redistribute | True | Solo aplica a Normal y Plus |

## Contexto: mixing overhead L

Para Ours++, cada palabra paga un costo extra de privacidad por el coin flip del mixing:

```
L = ln(max(p/(1-p), (1-p)/p))
```

| p | L |
|---|---|
| 0.5 | 0.000 |
| 0.6 | 0.405 |
| 0.7 | 0.847 |
| 0.8 | 1.386 |

Tanto `epsilon` como `s_epsilon` deben ser estrictamente mayores que L para que el método funcione.
Por ejemplo, con p=0.7, la combinación epsilon=1/s_epsilon=0.5 no es válida porque s_epsilon < L.

## Experimento 1: barrido de epsilon

Pregunta: curva privacidad-utilidad de los 3 métodos.

Fijar: seed=42, p=0.7, redistribute=True, s_epsilon=epsilon/2.

Variar: epsilon en {2, 4, 8, 16, 32}.
Se omite epsilon=1 porque s_epsilon=0.5 < L=0.847 para Plus con p=0.7.

```bash
cd Sensitive-Aware-DP-Text-Sanitization

# SanText (baseline)
for eps in 2 4 8 16 32; do
  python run_sanitizer.py \
    --data_dir ./datasets/i2b2/ \
    --method santext \
    --task i2b2 \
    --epsilon $eps --s_epsilon $eps \
    --seed 42 \
    --sensitive_words_file_path ./sensitive_mapping/flair_0.6_i2b2.json \
    --language en
done

# Ours (Normal) con redistribución
for eps in 2 4 8 16 32; do
  s_eps=$(python3 -c "print($eps / 2)")
  python run_sanitizer.py \
    --data_dir ./datasets/i2b2/ \
    --method normal \
    --task i2b2 \
    --epsilon $eps --s_epsilon $s_eps \
    --redistribute \
    --seed 42 \
    --sensitive_words_file_path ./sensitive_mapping/flair_0.6_i2b2.json \
    --language en
done

# Ours++ (Plus) con redistribución, p=0.7
for eps in 2 4 8 16 32; do
  s_eps=$(python3 -c "print($eps / 2)")
  python run_sanitizer.py \
    --data_dir ./datasets/i2b2/ \
    --method plus \
    --task i2b2 \
    --epsilon $eps --s_epsilon $s_eps \
    --p 0.7 \
    --redistribute \
    --seed 42 \
    --sensitive_words_file_path ./sensitive_mapping/flair_0.6_i2b2.json \
    --language en
done
```

## Experimento 2: efecto de la redistribución

Pregunta: redistribuir el presupuesto mejora la utilidad respecto a usar epsilon fijo?

Fijar: epsilon=8, s_epsilon=4, seed=42, p=0.7.

Variar: `--redistribute` vs `--no-redistribute` para Normal y Plus.

```bash
# Ours (Normal)
for redist in "--redistribute" "--no-redistribute"; do
  python run_sanitizer.py \
    --data_dir ./datasets/i2b2/ \
    --method normal \
    --task i2b2 \
    --epsilon 8 --s_epsilon 4 \
    $redist \
    --seed 42 \
    --sensitive_words_file_path ./sensitive_mapping/flair_0.6_i2b2.json \
    --language en
done

# Ours++ (Plus)
for redist in "--redistribute" "--no-redistribute"; do
  python run_sanitizer.py \
    --data_dir ./datasets/i2b2/ \
    --method plus \
    --task i2b2 \
    --epsilon 8 --s_epsilon 4 \
    --p 0.7 \
    $redist \
    --seed 42 \
    --sensitive_words_file_path ./sensitive_mapping/flair_0.6_i2b2.json \
    --language en
done
```

## Experimento 3: efecto de p

Pregunta: cuánto cuesta el mixing overhead a medida que p se aleja de 0.5?

Fijar: epsilon=8, s_epsilon=4, seed=42, redistribute=True, método=plus.

Variar: p en {0.5, 0.6, 0.7, 0.8}.

```bash
for p in 0.5 0.6 0.7 0.8; do
  python run_sanitizer.py \
    --data_dir ./datasets/i2b2/ \
    --method plus \
    --task i2b2 \
    --epsilon 8 --s_epsilon 4 \
    --p $p \
    --redistribute \
    --seed 42 \
    --sensitive_words_file_path ./sensitive_mapping/flair_0.6_i2b2.json \
    --language en
done
```

## Evaluación

Después de cada experimento, correr las métricas de calidad y el modelo downstream:

```bash
# Métricas de calidad (BERTScore, Jaccard, MAUVE, WMD)
python quality_metrics_task/run_quality_metrics.py

# Downstream NER
bash scripts/train_i2b2_ner.sh
```

## Qué reportar

Los resultados de sanitización quedan en `replacements_flair/` y las estadísticas en `corpus_statistics/`.

Para cada experimento, registrar:

| Métrica | Fuente |
|---------|--------|
| Epsilon total por documento (promedio) | `corpus_statistics/` |
| BERTScore | `quality_metrics_task/` |
| F1 downstream (NER) | salida de `train_i2b2_ner.sh` |

### Tabla del Experimento 1 (barrido de epsilon)

| epsilon | SanText F1 | Ours F1 | Ours++ F1 | SanText BERTScore | Ours BERTScore | Ours++ BERTScore |
|---------|-----------|---------|-----------|-------------------|---------------|-----------------|
| 2 | | | | | | |
| 4 | | | | | | |
| 8 | | | | | | |
| 16 | | | | | | |
| 32 | | | | | | |

### Tabla del Experimento 2 (redistribución)

| Método | Redistribuir | F1 | BERTScore | Epsilon total promedio |
|--------|-------------|----|-----------|-----------------------|
| Ours | Sí | | | |
| Ours | No | | | |
| Ours++ | Sí | | | |
| Ours++ | No | | | |

### Tabla del Experimento 3 (efecto de p)

| p | L | F1 | BERTScore | Epsilon total promedio |
|---|---|----|-----------|-----------------------|
| 0.5 | 0.000 | | | |
| 0.6 | 0.405 | | | |
| 0.7 | 0.847 | | | |
| 0.8 | 1.386 | | | |

## Resultados finales (para el paper)

Una vez identificadas las configuraciones interesantes, repetir con los 5 seeds
(1, 21, 42, 84, 132) y reportar media +/- desviación estándar.
