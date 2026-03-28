<div align="center">

# 🫁 PneumoScan — Microservicios en Amazon EKS

### Práctica 5: Migración a Arquitectura de Microservicios

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=flat&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-EKS-326CE5?style=flat&logo=kubernetes&logoColor=white)](https://aws.amazon.com/eks/)
[![AWS](https://img.shields.io/badge/AWS-ECR%20%2B%20EKS-FF9900?style=flat&logo=amazon-aws&logoColor=white)](https://aws.amazon.com)

🌐 **Demo en vivo:** http://ad464dc9620d6408bab2b9f96b872e3b-437194755.us-east-1.elb.amazonaws.com

</div>

---

## ¿Qué es PneumoScan?

PneumoScan es un sistema de apoyo al diagnóstico médico que analiza radiografías de tórax con una **Red Neuronal Convolucional (CNN)** y clasifica la imagen en tres categorías: Neumonía Bacteriana, Neumonía Viral o Pulmón Normal. Genera además un mapa de calor **Grad-CAM** que indica qué zonas de la radiografía influyeron en el diagnóstico.

Esta versión migra la aplicación de un monolito Docker a **3 microservicios independientes** desplegados en **Amazon EKS** con alta disponibilidad (2 réplicas por pod).

---

## Índice

1. [Arquitectura de microservicios](#arquitectura-de-microservicios)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Prueba local con Docker Compose](#prueba-local-con-docker-compose)
4. [Despliegue en Amazon EKS](#despliegue-en-amazon-eks)
5. [Manifiestos Kubernetes](#manifiestos-kubernetes)
6. [API de los microservicios](#api-de-los-microservicios)
7. [Variables de entorno](#variables-de-entorno)

---

## Arquitectura de microservicios

### Los 3 microservicios

| Microservicio | Tecnología | Réplicas | Puerto | Responsabilidad |
|---|---|---|---|---|
| `frontend` | Flask 3.0, psycopg2, requests | 2 | 80 | UI, llamadas HTTP al ai-service, persistencia en PostgreSQL |
| `ai-service` | Flask 3.0, TensorFlow 2.20, OpenCV | 2 | 8000 | Inferencia CNN, Grad-CAM, devuelve heatmap en base64 |
| `db` | PostgreSQL 15 | 1 | 5432 | Almacena historial de predicciones (PVC EBS 10Gi) |

### Diagrama de arquitectura

```
                          INTERNET
                              │
                              ▼
              ┌───────────────────────────────┐
              │   AWS Classic Load Balancer    │
              │   (puerto 80, internet-facing) │
              └──────────┬────────────────────┘
                         │
              ┌──────────▼──────────────────────────┐
              │         EKS Cluster                  │
              │     pneumoscan-eks (us-east-1)        │
              │     3 nodos t3.small (2vCPU, 2GB)    │
              │                                      │
              │   ┌─────────────────────────────┐   │
              │   │   namespace: pneumoscan      │   │
              │   │                             │   │
              │   │  ┌──────────┐  ┌──────────┐ │   │
              │   │  │frontend-1│  │frontend-2│ │   │  ← 2 réplicas
              │   │  │ :80      │  │ :80      │ │   │
              │   │  └────┬─────┘  └────┬─────┘ │   │
              │   │       └──────┬───────┘       │   │
              │   │    HTTP POST /predict         │   │
              │   │  ┌─────────────────────────┐ │   │
              │   │  │  ai-service ×2 (:8000)  │ │   │  ← ClusterIP
              │   │  │  Flask + TensorFlow      │ │   │
              │   │  └─────────────────────────┘ │   │
              │   │                             │   │
              │   │  ┌─────────────────────────┐ │   │
              │   │  │  db StatefulSet (:5432)  │ │   │  ← EBS 10Gi
              │   │  │  PostgreSQL 15           │ │   │
              │   │  └─────────────────────────┘ │   │
              │   └─────────────────────────────┘   │
              └──────────────────────────────────────┘
```

### Flujo de una predicción

```
1. Usuario sube radiografía → frontend (puerto 80)
2. frontend → POST /predict con imagen → ai-service (ClusterIP :8000)
3. ai-service:
   a. Preprocesa imagen (resize 512x512, CLAHE, normalizar)
   b. Inferencia CNN → label (bacteriana/normal/viral) + probability
   c. Grad-CAM → heatmap PNG codificado en base64
   d. Retorna JSON: {label, probability, heatmap_base64}
4. frontend:
   a. Decodifica heatmap_base64 → archivo en /tmp/heatmaps/
   b. Guarda en PostgreSQL → tabla predicciones
   c. Renderiza resultado en HTML
```

---

## Estructura del repositorio

```
uaoneumonia/
│
├── services/                       ← Microservicios
│   ├── frontend/
│   │   ├── app.py                  ← Flask sin TensorFlow
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── templates/
│   │       └── index.html
│   └── ai-service/
│       ├── app.py                  ← Flask + TensorFlow, expone /predict /health
│       ├── Dockerfile              ← modelo conv_MLP_84.h5 baked en imagen
│       ├── requirements.txt
│       └── src/                    ← módulos ML (integrator, load_model, grad_cam...)
│
├── k8s/                            ← Manifiestos Kubernetes
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── frontend/
│   │   ├── deployment.yaml         ← 2 réplicas, RollingUpdate, /health probes
│   │   └── service.yaml            ← type: LoadBalancer, puerto 80
│   ├── ai-service/
│   │   ├── deployment.yaml         ← 2 réplicas, initialDelay 60s (TF load)
│   │   └── service.yaml            ← type: ClusterIP, puerto 8000
│   └── db/
│       ├── statefulset.yaml        ← postgres:15, PVC volumeClaimTemplate
│       └── service.yaml            ← type: ClusterIP, puerto 5432
│
├── docker-compose.dev.yml          ← Prueba local de los 3 microservicios
└── Vagrantfile                     ← VM Ubuntu con AWS CLI, kubectl, eksctl
```

---

## Prueba local con Docker Compose

Levanta los 3 microservicios localmente antes de desplegar en EKS.

### Prerequisitos

- Docker Desktop instalado y corriendo
- El modelo en `models/conv_MLP_84.h5` (raíz del proyecto)

### Pasos

```bash
# Clonar el repositorio (rama microservicios)
git clone -b feature/microservices-eks https://github.com/L4M4rck/Detector-de-Neumonia---AWS.git
cd Detector-de-Neumonia---AWS

# Levantar los 3 servicios
docker compose -f docker-compose.dev.yml up --build
```

Esperar ~2 minutos (TensorFlow tarda en cargar el modelo). Cuando aparezca:
```
pneumoscan_ai | Listening at: http://0.0.0.0:8000
```

Abrir en el navegador: **http://localhost**

```bash
# Verificar salud de los servicios
curl http://localhost/health          # frontend → {"status": "ok"}
curl http://localhost:8000/health     # ai-service → {"status": "ok", "service": "ai-service"}

# Ver logs del ai-service
docker compose -f docker-compose.dev.yml logs -f ai-service

# Detener
docker compose -f docker-compose.dev.yml down
```

---

## Despliegue en Amazon EKS

### Prerequisitos

- AWS CLI v2 instalado y configurado (`aws configure`)
- kubectl v1.31+
- eksctl v0.200+
- Docker Desktop

### Paso 1 — Crear repositorios ECR

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

aws ecr create-repository --repository-name pneumoscan-frontend --region $REGION
aws ecr create-repository --repository-name pneumoscan-ai --region $REGION

# Login a ECR
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
```

### Paso 2 — Construir y subir imágenes

```bash
# Copiar el modelo al contexto del ai-service
mkdir -p services/ai-service/models
cp models/conv_MLP_84.h5 services/ai-service/models/

# Build y push frontend
docker build -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/pneumoscan-frontend:latest services/frontend/
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/pneumoscan-frontend:latest

# Build y push ai-service (~5 min por TensorFlow)
docker build -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/pneumoscan-ai:latest services/ai-service/
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/pneumoscan-ai:latest
```

### Paso 3 — Crear el cluster EKS

> **Nota AWS Academy:** Solo son elegibles instancias Free Tier. Usar `t3.small` (2 vCPU, 2 GB RAM). No usar t3.medium ni t3.large.

```bash
eksctl create cluster \
  --name pneumoscan-eks \
  --region us-east-1 \
  --version 1.34 \
  --nodegroup-name pneumoscan-nodes \
  --node-type t3.small \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 4 \
  --managed
```

El proceso tarda ~15 minutos. eksctl configura `kubectl` automáticamente.

> **Windows:** Si `kubectl` dice `executable aws not found`, editar `~/.kube/config` y cambiar `command: aws` por la ruta completa: `command: C:\Program Files\Amazon\AWSCLIV2\aws.exe`

### Paso 4 — Instalar EBS CSI Driver

El driver EBS CSI es necesario para que PostgreSQL pueda usar volúmenes persistentes.

```bash
# Instalar el addon
aws eks create-addon \
  --cluster-name pneumoscan-eks \
  --addon-name aws-ebs-csi-driver \
  --region us-east-1

# Obtener el nombre del NodeInstanceRole
NODE_ROLE=$(aws iam list-roles \
  --query "Roles[?contains(RoleName,'NodeInstanceRole')].RoleName" \
  --output text)

# Adjuntar la política IAM necesaria
aws iam attach-role-policy \
  --role-name $NODE_ROLE \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy

# Reiniciar el controlador para tomar los nuevos permisos
kubectl rollout restart deployment/ebs-csi-controller -n kube-system

# Verificar que el controlador está Running
kubectl get pods -n kube-system -l app=ebs-csi-controller
```

### Paso 5 — Desplegar en Kubernetes

```bash
# Recursos base
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Microservicios
kubectl apply -f k8s/db/
kubectl apply -f k8s/ai-service/
kubectl apply -f k8s/frontend/
```

### Paso 6 — Verificar el despliegue

```bash
# Ver todos los pods (esperar ~2 min)
kubectl get pods -n pneumoscan -w

# Salida esperada:
# NAME                          READY   STATUS    RESTARTS   AGE
# ai-service-xxx-yyy   1/1   Running   0   3m
# ai-service-xxx-zzz   1/1   Running   0   3m
# db-0                 1/1   Running   0   3m
# frontend-xxx-yyy     1/1   Running   0   3m
# frontend-xxx-zzz     1/1   Running   0   3m

# Obtener URL pública del LoadBalancer
kubectl get service frontend-service -n pneumoscan
# → EXTERNAL-IP: <nombre>.us-east-1.elb.amazonaws.com

# Ver logs del ai-service (verificar que TF cargó el modelo)
kubectl logs -n pneumoscan -l app=ai-service --tail=20
```

### Limpieza (evitar costos)

```bash
# Eliminar todos los recursos de Kubernetes
kubectl delete namespace pneumoscan

# Eliminar el cluster EKS
eksctl delete cluster --name pneumoscan-eks --region us-east-1
```

---

## Manifiestos Kubernetes

### ConfigMap (`k8s/configmap.yaml`)

Variables de entorno compartidas entre todos los pods:

| Variable | Valor | Descripción |
|---|---|---|
| `AI_SERVICE_URL` | `http://ai-service:8000` | URL interna del ai-service |
| `DB_HOST` | `db-service` | Nombre del servicio de base de datos |
| `DB_PORT` | `5432` | Puerto PostgreSQL |
| `DB_NAME` | `neumonia` | Nombre de la base de datos |
| `DB_USER` | `neumonia_user` | Usuario de la base de datos |
| `MODEL_PATH` | `/app/models/conv_MLP_84.h5` | Ruta del modelo en el contenedor |

### Secret (`k8s/secret.yaml`)

```bash
# Crear un secret con contraseña segura (reemplazar en producción)
kubectl create secret generic pneumoscan-secrets \
  --from-literal=DB_PASSWORD=tu_password_seguro \
  --from-literal=POSTGRES_PASSWORD=tu_password_seguro \
  -n pneumoscan
```

### Decisiones de diseño

| Decisión | Motivo |
|---|---|
| Modelo **baked en la imagen** del ai-service | Evita configurar EFS CSI driver (complejidad innecesaria). Imagen ~1.5 GB. |
| Gunicorn con **1 worker sync, sin --preload** | TensorFlow no es fork-safe. El modelo carga dentro del worker, después del fork. |
| `initialDelaySeconds: 60/90` en ai-service | TF tarda 30-60s en cargar el modelo; sin delay los probes matarían el pod. |
| `RollingUpdate` con `maxUnavailable: 0` | Zero-downtime: siempre hay al menos 2 pods sirviendo tráfico. |
| PostgreSQL como **StatefulSet** con PVC EBS | Garantiza identidad estable y datos persistentes entre reinicios. |

---

## API de los microservicios

### frontend (puerto 80)

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Interfaz web principal |
| `POST` | `/` | Enviar imagen para análisis |
| `GET` | `/health` | Health check (para Load Balancer) |
| `GET` | `/uploads/<filename>` | Servir imagen subida |
| `GET` | `/heatmaps/<filename>` | Servir mapa de calor |
| `GET` | `/export-pdf` | Descargar reporte en PDF |
| `GET` | `/export-csv` | Descargar historial en CSV |

### ai-service (puerto 8000, ClusterIP interno)

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/predict` | Recibe imagen (multipart), retorna JSON con diagnóstico y heatmap |
| `GET` | `/health` | Health check del servicio de IA |

**Respuesta de `/predict`:**

```json
{
  "label": "bacteriana",
  "probability": 94.27,
  "heatmap_base64": "iVBORw0KGgoAAAA..."
}
```

---

## Variables de entorno

| Variable | Servicio | Descripción |
|---|---|---|
| `AI_SERVICE_URL` | frontend | URL del ai-service (`http://ai-service:8000`) |
| `DB_HOST` | frontend | Host de PostgreSQL |
| `DB_PORT` | frontend | Puerto de PostgreSQL (`5432`) |
| `DB_NAME` | frontend, db | Nombre de la base de datos |
| `DB_USER` | frontend, db | Usuario de PostgreSQL |
| `DB_PASSWORD` | frontend | Contraseña de PostgreSQL |
| `POSTGRES_PASSWORD` | db | Contraseña para el contenedor PostgreSQL |
| `MODEL_PATH` | ai-service | Ruta del modelo `.h5` dentro del contenedor |

---
### Pods
![WhatsApp Image 2026-03-27 at 8 42 55 PM](https://github.com/user-attachments/assets/1f21b9e3-1b13-43d7-b858-501efb3776c2)
---
### Nodos
![Nodos](https://github.com/user-attachments/assets/53c9ac3f-cd88-4fbc-ab02-ff3a538a0ece)
---
### Replicas
![Replicas](https://github.com/user-attachments/assets/c9696684-6015-4f16-8265-6ee8d2582b96)
---
<div align="center">

**Aviso médico**

Este sistema es una herramienta de apoyo al diagnóstico basada en inteligencia artificial.
No reemplaza el criterio clínico de un médico especialista.

---

Desarrollado como parte de la asignatura **Computación en la Nube**
Universidad Autónoma de Occidente — Práctica 5

</div>
