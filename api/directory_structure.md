# API Directory Structure

```
├── .gitkeep
├── config/
│   └── config.go
├── go.mod
├── go.sum
├── internal/
│   ├── handler/
│   │   ├── place/
│   │   │   ├── handler.go
│   │   │   └── routes.go
│   │   └── util/
│   │       └── init.go
│   ├── model/
│   │   └── entity/
│   │       ├── places.go
│   │       └── user.go
│   ├── repository/
│   │   ├── ml-api/
│   │   │   ├── init.go
│   │   │   └── ml-client.go
│   │   ├── ml-api.go
│   │   ├── places.go
│   │   ├── postgres/
│   │   │   ├── init.go
│   │   │   └── places.go
│   │   └── util/
│   │       └── init.go
│   └── service/
│       ├── place/
│       │   ├── impl.go
│       │   └── init.go
│       ├── place.go
│       └── util/
│           └── init.go
└── main.go
```

## Structure Overview

- **config/**: Configuration files
- **internal/**: Core application code
  - **handler/**: HTTP handlers/controllers
  - **model/**: Data models/entities
  - **repository/**: Data access layer
  - **service/**: Business logic layer
- **Root files**: go.mod, go.sum, main.go