# Build Notes

## DeepSeek OCR Rust Server Build

### Build Time
- **First build**: 10-15 minutes (compiles all Rust dependencies)
- **Subsequent builds**: ~1-2 minutes (uses Docker layer cache)

### Rust Version Requirement
- **Required**: Rust nightly
- **Reason**: The deepseek-ocr.rs project uses Rust edition 2024, which is only available in nightly builds
- **Note**: README says "Rust 1.78+" but edition 2024 actually requires nightly toolchain

### Build Command
```bash
docker compose -f docker-compose.app.yml build deepseek-ocr-server
```

### Common Build Issues

#### 1. "feature `edition2024` is required"
**Problem**: Rust version too old  
**Solution**: Update Dockerfile to use Rust 1.83.0 or newer

#### 2. CUDA toolkit not found
**Problem**: Wrong base image  
**Solution**: Use `nvidia/cuda:12.2.0-devel-ubuntu22.04` for building

#### 3. Out of memory during build
**Problem**: Rust compilation uses lots of RAM  
**Solution**: Increase Docker memory limit to 8GB+

#### 4. Build takes too long
**Expected**: 10-15 minutes on first build is normal  
**Tip**: Use `--progress=plain` to see detailed build output

### Build Stages

The Dockerfile uses multi-stage build:

1. **Builder stage** (`nvidia/cuda:12.2.0-devel-ubuntu22.04`)
   - Installs Rust 1.83.0
   - Compiles deepseek-ocr-server with CUDA support
   - ~12GB temporary disk space needed

2. **Runtime stage** (`nvidia/cuda:12.2.0-runtime-ubuntu22.04`)
   - Minimal runtime environment
   - Only ~2GB final image size
   - Includes CUDA runtime libraries

### Monitoring Build Progress

```bash
# See detailed build output
docker compose -f docker-compose.app.yml build --progress=plain deepseek-ocr-server

# Check build logs
docker compose -f docker-compose.app.yml logs deepseek-ocr-server

# Verify the binary was built
docker compose -f docker-compose.app.yml run --rm deepseek-ocr-server --help
```

### Build Cache

Docker caches each layer:
- Rust installation: cached after first build
- Dependencies download: cached after first build  
- Source compilation: rebuilds only if source changes

To force rebuild without cache:
```bash
docker compose -f docker-compose.app.yml build --no-cache deepseek-ocr-server
```

### Disk Space Requirements

| Stage | Space Needed |
|-------|--------------|
| CUDA base images | ~4GB |
| Rust toolchain | ~1.5GB |
| Cargo build artifacts | ~8GB |
| Final image | ~2GB |
| **Total** | **~15GB** |

Clean up build cache:
```bash
docker builder prune -a
```

### Expected Build Output

Successful build will show:
```
=> [builder 7/7] RUN cargo build --release -p deepseek-ocr-server --features cuda
   Compiling 200+ crates...
   Finished release [optimized] target(s) in 8m 32s

=> [stage-1 4/6] COPY --from=builder /build/target/release/deepseek-ocr-server
=> [stage-1 5/6] COPY entrypoint.sh /app/entrypoint.sh
=> [stage-1 6/6] RUN chmod +x /app/entrypoint.sh
=> exporting to image
=> => naming to docker.io/library/documents-service-deepseek-ocr-server
```

### Verification

After build completes, verify:

```bash
# Check image exists
docker images | grep deepseek-ocr-server

# Test binary
docker compose -f docker-compose.app.yml run --rm deepseek-ocr-server --help

# Start server
docker compose -f docker-compose.app.yml up deepseek-ocr-server
```

### Troubleshooting Failed Builds

If build fails:

1. **Check Docker resources**:
   ```bash
   docker system df
   ```

2. **Clean up old builds**:
   ```bash
   docker system prune -a --volumes
   ```

3. **Verify NVIDIA toolkit** (if using GPU):
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
   ```

4. **Check Rust version in builder**:
   ```bash
   docker compose -f docker-compose.app.yml build --no-cache \
     --build-arg RUST_VERSION=1.83.0 deepseek-ocr-server
   ```

## Related Documentation

- [DeepSeek OCR Rust Setup](./DEEPSEEK_OCR_RUST_SETUP.md)
- [Pipeline Architecture](./PIPELINE_ARCHITECTURE.md)
