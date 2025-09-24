// 与您的Java背景结合
@RestController
@RequestMapping("/api/rs")
public class RemoteSensingService {
    
    @Autowired
    private PythonModelService pythonService;
    
    @PostMapping("/analyze")
    public ResponseEntity<AnalysisResult> analyzeImage(
        @RequestParam("file") MultipartFile file,
        @RequestParam("model") String modelType,
        @RequestParam("tasks") List<String> tasks
    ) {
        // 调用Python模型
        byte[] imageData = file.getBytes();
        
        // 使用gRPC或REST调用Python服务
        AnalysisResult result = pythonService.analyze(
            imageData, modelType, tasks
        );
        
        return ResponseEntity.ok(result);
    }
}