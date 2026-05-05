/**
 * ============================================================================
 * YEMOT HAMASHIACH AI ROUTER - ADVANCED SYSTEM
 * Model: gemini-3.1-flash-lite-preview
 * Deployment: Vercel Serverless Functions
 * Architecture: RAG-Lite (Local TF-IDF Search + AI Extraction)
 * Features: Multi-Key Load Balancing, Zero-Cost Routing (Nitoviya)
 * ============================================================================
 */

const express = require('express');
const fs = require('fs');
const path = require('path');
const https = require('https');

// ============================================================================
// 1. UTILITIES & ADVANCED DATA STRUCTURES
// ============================================================================

/**
 * Advanced Logger to keep track of Vercel serverless execution
 */
class Logger {
    static info(msg, data = '') { console.log(`[INFO] ${new Date().toISOString()} - ${msg}`, data); }
    static error(msg, err = '') { console.error(`[ERROR] ${new Date().toISOString()} - ${msg}`, err); }
    static debug(msg, data = '') { console.debug(`[DEBUG] ${new Date().toISOString()} - ${msg}`, data); }
}

/**
 * Memory Cache to maintain state between Yemot API calls since Vercel is stateless.
 * We will also use Yemot's query parameters to pass state to ensure 100% reliability.
 */
class StateManager {
    static encodeState(obj) {
        return Buffer.from(JSON.stringify(obj)).toString('base64');
    }
    static decodeState(str) {
        try {
            return JSON.parse(Buffer.from(str, 'base64').toString('utf8'));
        } catch (e) {
            return {};
        }
    }
}

// ============================================================================
// 2. GEMINI MULTI-KEY LOAD BALANCER & RATE LIMIT MANAGER
// ============================================================================

/**
 * Manages an array of Gemini API keys to bypass rate limits gracefully.
 * Implements Round-Robin and Error Tracking.
 */
class GeminiKeyManager {
    constructor(keysString) {
        if (!keysString) throw new Error("No Gemini keys provided.");
        this.keys = keysString.split(',').map(k => k.trim()).filter(k => k.length > 0);
        this.keyStats = this.keys.map(key => ({
            key: key,
            failures: 0,
            lastUsed: 0,
            cooldownUntil: 0
        }));
        Logger.info(`Initialized Gemini Key Manager with ${this.keys.length} keys.`);
    }

    /**
     * Gets the best available key, considering cooldowns and failure rates.
     */
    getBestKey() {
        const now = Date.now();
        // Filter keys that are not in cooldown
        let availableKeys = this.keyStats.filter(k => k.cooldownUntil < now);
        
        if (availableKeys.length === 0) {
            Logger.error("All Gemini keys are currently in cooldown! Forcing fallback to the key with lowest cooldown.");
            // Fallback: pick the one that will be ready soonest
            availableKeys = [...this.keyStats].sort((a, b) => a.cooldownUntil - b.cooldownUntil);
        }

        // Sort by least failures, then by least recently used
        availableKeys.sort((a, b) => {
            if (a.failures !== b.failures) return a.failures - b.failures;
            return a.lastUsed - b.lastUsed;
        });

        const selected = availableKeys[0];
        selected.lastUsed = now;
        return selected.key;
    }

    /**
     * Reports a failure for a key, applying exponential backoff for cooldowns.
     */
    reportFailure(keyStr) {
        const keyObj = this.keyStats.find(k => k.key === keyStr);
        if (keyObj) {
            keyObj.failures += 1;
            // Cooldown: 10s for first fail, 30s, 60s...
            const cooldownPenalty = Math.min(10000 * Math.pow(2, keyObj.failures - 1), 300000); 
            keyObj.cooldownUntil = Date.now() + cooldownPenalty;
            Logger.error(`Key starting with ${keyStr.substring(0, 5)}... reported failure. Cooldown for ${cooldownPenalty/1000}s.`);
        }
    }

    reportSuccess(keyStr) {
        const keyObj = this.keyStats.find(k => k.key === keyStr);
        if (keyObj) {
            keyObj.failures = 0; // Reset failures on success
        }
    }
}

// ============================================================================
// 3. HEBREW NLP & ADVANCED LOCAL SEARCH ENGINE
// ============================================================================

/**
 * Built-in Hebrew Natural Language Processing
 * Prevents wasting tokens by doing smart local search.
 */
class HebrewNLP {
    static stopWords = new Set([
        "אני", "רוצה", "מחפש", "קו", "של", "את", "בבקשה", "יש", "לכם", "אולי",
        "לעבור", "לשמוע", "משהו", "עם", "הרבה", "בעיקר", "תביא", "לי", "כזה",
        "שיהיה", "בו", "על", "אל", "או", "גם", "כן", "לא", "אבל"
    ]);

    /**
     * Tokenizes and cleans Hebrew text
     */
    static tokenize(text) {
        if (!text) return[];
        // Remove punctuation and split by whitespace
        const words = text.replace(/[^\u0590-\u05FF\s]/g, '').split(/\s+/);
        return words
            .filter(w => w.length > 1) // Remove single letters
            .filter(w => !this.stopWords.has(w));
    }

    /**
     * Calculates Levenshtein Distance (Fuzzy Matching for typos in Hebrew transcription)
     */
    static levenshtein(a, b) {
        const matrix =[];
        for (let i = 0; i <= b.length; i++) matrix[i] = [i];
        for (let j = 0; j <= a.length; j++) matrix[0][j] = j;

        for (let i = 1; i <= b.length; i++) {
            for (let j = 1; j <= a.length; j++) {
                if (b.charAt(i - 1) === a.charAt(j - 1)) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(
                        matrix[i - 1][j - 1] + 1, // substitution
                        Math.min(matrix[i][j - 1] + 1, // insertion
                        matrix[i - 1][j] + 1) // deletion
                    );
                }
            }
        }
        return matrix[b.length][a.length];
    }
}

/**
 * The Local Search Engine that scans lines.txt
 */
class LocalSearchEngine {
    constructor(filePath) {
        this.linesDb =[];
        this.loadDatabase(filePath);
    }

    loadDatabase(filePath) {
        try {
            const absolutePath = path.join(__dirname, filePath);
            const data = fs.readFileSync(absolutePath, 'utf8');
            const rows = data.split('\n');
            
            for (const row of rows) {
                if (!row.trim()) continue;
                const parts = row.split('|');
                if (parts.length >= 3) {
                    const phone = parts[0].trim();
                    const name = parts[1].trim();
                    const desc = parts[2].trim();
                    
                    // Pre-tokenize description for faster search
                    const tokens = HebrewNLP.tokenize(name + " " + desc);
                    
                    this.linesDb.push({ phone, name, desc, tokens });
                }
            }
            Logger.info(`Loaded ${this.linesDb.length} IVR lines into local search engine.`);
        } catch (error) {
            Logger.error("Failed to load lines.txt database", error);
        }
    }

    /**
     * Scores records against extracted keywords using overlap & fuzzy match
     */
    search(queryKeywords, topK = 3) {
        if (!queryKeywords || queryKeywords.length === 0) return[];

        const scoredLines = this.linesDb.map(record => {
            let score = 0;
            
            for (const qWord of queryKeywords) {
                let bestWordScore = 0;
                
                for (const rWord of record.tokens) {
                    // Exact match gets high score
                    if (rWord === qWord) {
                        bestWordScore = Math.max(bestWordScore, 10);
                    } else {
                        // Fuzzy match check (allow 1 typo for words > 4 chars)
                        if (qWord.length > 4 && rWord.length > 4) {
                            const dist = HebrewNLP.levenshtein(qWord, rWord);
                            if (dist === 1) bestWordScore = Math.max(bestWordScore, 5);
                            if (dist === 2 && qWord.length > 6) bestWordScore = Math.max(bestWordScore, 3);
                        }
                    }
                }
                score += bestWordScore;
            }
            
            return { ...record, score };
        });

        // Filter lines that have at least some relevance, sort by score descending
        const results = scoredLines
            .filter(r => r.score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, topK);

        Logger.info(`Search found ${results.length} matches for keywords: ${queryKeywords.join(',')}`);
        return results;
    }
}

// ============================================================================
// 4. YEMOT HAMASHIACH API CLIENT
// ============================================================================

/**
 * Handles complex communication with Yemot Hamashiach servers.
 */
class YemotAPIClient {
    constructor(tokenString) {
        this.baseUrl = "https://www.call2all.co.il/ym/api";
        this.token = tokenString; // Format: 0771234567:123456
    }

    /**
     * Downloads an audio file recorded by the user.
     * Uses native HTTPS to handle binary streams efficiently.
     */
    async downloadAudio(filePath) {
        return new Promise((resolve, reject) => {
            const url = `${this.baseUrl}/DownloadFile?token=${encodeURIComponent(this.token)}&path=${encodeURIComponent(filePath)}`;
            Logger.debug(`Downloading audio from Yemot: ${filePath}`);
            
            https.get(url, (res) => {
                if (res.statusCode !== 200) {
                    reject(new Error(`Failed to download audio, status code: ${res.statusCode}`));
                    return;
                }

                const chunks =[];
                res.on('data', (chunk) => chunks.push(chunk));
                res.on('end', () => {
                    const buffer = Buffer.concat(chunks);
                    Logger.info(`Successfully downloaded audio, size: ${buffer.length} bytes`);
                    resolve(buffer);
                });
            }).on('error', (err) => {
                reject(err);
            });
        });
    }

    /**
     * Updates an extension via API to create the dynamic Nitoviya routing.
     * ZERO COST ROUTING AS REQUESTED!
     */
    async setupNitoviyaRouting(extensionPath, destinationNumber) {
        Logger.info(`Setting up Nitoviya zero-cost routing at ${extensionPath} to ${destinationNumber}`);
        
        // We build the extension configuration string
        const extConfig = `type=nitoviya\nnitoviya_dial_to=${destinationNumber}\n`;
        
        const url = `${this.baseUrl}/UploadTextFile?token=${encodeURIComponent(this.token)}&what=ivr2:${encodeURIComponent(extensionPath)}/ext.ini&contents=${encodeURIComponent(extConfig)}`;

        return new Promise((resolve, reject) => {
            https.get(url, (res) => {
                let data = '';
                res.on('data', (chunk) => data += chunk);
                res.on('end', () => {
                    try {
                        const json = JSON.parse(data);
                        if (json.responseStatus === 'OK') {
                            Logger.info(`Successfully configured Nitoviya extension at ${extensionPath}`);
                            resolve(true);
                        } else {
                            Logger.error(`Failed to configure Nitoviya: ${data}`);
                            resolve(false);
                        }
                    } catch (e) {
                        Logger.error("Failed to parse Yemot response", e);
                        resolve(false);
                    }
                });
            }).on('error', (err) => reject(err));
        });
    }
}

// ============================================================================
// 5. GEMINI AI CLIENT (STRICTLY gemini-3.1-flash-lite-preview)
// ============================================================================

/**
 * Handles communication with Google Gemini API.
 */
class GeminiClient {
    constructor(keyManager) {
        this.keyManager = keyManager;
        // The ONLY allowed model as per strict user instruction
        this.model = "gemini-3.1-flash-lite-preview"; 
    }

    /**
     * Sends Audio to Gemini for transcription and intent extraction.
     */
    async processAudioRequest(audioBuffer) {
        const apiKey = this.keyManager.getBestKey();
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${this.model}:generateContent?key=${apiKey}`;

        // Convert audio buffer to base64 for inlineData payload
        const base64Audio = audioBuffer.toString('base64');

        const prompt = `
        אתה מערכת AI המנתחת בקשות קוליות של מתקשרים בטלפון.
        האזן להקלטה המצורפת, תמלל אותה, ומצא מה המתקשר מחפש.
        לאחר מכן, חלץ מילות מפתח קריטיות לחיפוש (למשל סוג המוזיקה, נושא השיעור וכו').
        
        חובה להחזיר את התשובה *אך ורק* כפורמט JSON תקין, ללא שום טקסט נוסף לפני או אחרי, במבנה הבא:
        {
            "transcription": "התמלול המלא של ההקלטה",
            "keywords":["מילת_מפתח_1", "מילת_מפתח_2"]
        }
        `;

        const payload = {
            contents: [
                {
                    parts:[
                        { text: prompt },
                        {
                            inlineData: {
                                mimeType: "audio/wav", // Yemot records in WAV
                                data: base64Audio
                            }
                        }
                    ]
                }
            ],
            generationConfig: {
                temperature: 0.2, // Low temp for deterministic extraction
                responseMimeType: "application/json"
            }
        };

        return new Promise((resolve, reject) => {
            const req = https.request(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            }, (res) => {
                let responseData = '';
                res.on('data', (chunk) => responseData += chunk);
                res.on('end', () => {
                    if (res.statusCode === 200) {
                        this.keyManager.reportSuccess(apiKey);
                        try {
                            const json = JSON.parse(responseData);
                            const content = json.candidates[0].content.parts[0].text;
                            // Clean markdown JSON formatting if Gemini adds it
                            const cleanContent = content.replace(/```json/g, '').replace(/```/g, '').trim();
                            const resultObj = JSON.parse(cleanContent);
                            Logger.info("Gemini processed audio successfully", resultObj);
                            resolve(resultObj);
                        } catch (e) {
                            Logger.error("Failed to parse Gemini JSON output", e);
                            reject(new Error("Invalid JSON from Gemini"));
                        }
                    } else if (res.statusCode === 429) {
                        this.keyManager.reportFailure(apiKey);
                        Logger.error("Gemini Rate Limit hit (429).");
                        reject(new Error("RATE_LIMIT"));
                    } else {
                        this.keyManager.reportFailure(apiKey);
                        Logger.error(`Gemini API Error: ${res.statusCode}`, responseData);
                        reject(new Error(`API_ERROR_${res.statusCode}`));
                    }
                });
            });

            req.on('error', (err) => {
                this.keyManager.reportFailure(apiKey);
                reject(err);
            });

            req.write(JSON.stringify(payload));
            req.end();
        });
    }

    /**
     * Wrapper with automatic retry logic across different keys.
     */
    async safeProcessAudio(audioBuffer, retries = 3) {
        for (let i = 0; i < retries; i++) {
            try {
                return await this.processAudioRequest(audioBuffer);
            } catch (error) {
                if (error.message === "RATE_LIMIT" || error.message.startsWith("API_ERROR")) {
                    Logger.info(`Retrying Gemini request (${i + 1}/${retries})...`);
                    // Small delay before retry
                    await new Promise(r => setTimeout(r, 1000));
                } else {
                    throw error; // Other parsing errors, don't retry blindly
                }
            }
        }
        throw new Error("Failed to process audio after multiple retries across keys.");
    }
}

// ============================================================================
// 6. YEMOT IVR STATE MACHINE & RESPONSE BUILDER
// ============================================================================

/**
 * Builds the specific strings Yemot Hamashiach expects to receive.
 */
class YemotResponseBuilder {
    constructor() {
        this.commands =[];
    }

    addRead(textToSay, variableName, options = "record_audio,,record") {
        // Options usually format: parameterName,ask_ok,min,max,type,block_asterisk
        this.commands.push(`read=t-${textToSay}=${variableName}`);
    }

    addGoToFolder(folderPath) {
        this.commands.push(`go_to_folder=${folderPath}`);
    }

    addIdListMessage(textToSay) {
        this.commands.push(`id_list_message=t-${textToSay}`);
    }

    toString() {
        return this.commands.join('&');
    }
}

// Initialize Global Search Engine
const searchEngine = new LocalSearchEngine('lines.txt');

// ============================================================================
// 7. EXPRESS APPLICATION & WEBHOOK ROUTING
// ============================================================================

const app = express();
// Yemot sends data via URL encoded queries or form data
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

/**
 * Main Webhook Endpoint for Yemot API
 */
app.all('*', async (req, res) => {
    Logger.info("Received request from Yemot");
    
    try {
        // 1. Extract configurations from the query parameters
        // The user configures api_add_0=yemot_token=... and api_add_1=gemini_keys=... in ext.ini
        const queryParams = req.method === 'POST' ? req.body : req.query;
        
        const yemotToken = queryParams.yemot_token;
        const geminiKeysString = queryParams.gemini_keys;
        
        if (!yemotToken || !geminiKeysString) {
            Logger.error("Missing critical authentication parameters from Yemot extensions.");
            return res.send("id_list_message=t-שגיאת מערכת. חסרים נתוני התחברות בהגדרות השלוחה.&go_to_folder=hangup");
        }

        const yemotApi = new YemotAPIClient(yemotToken);
        const keyManager = new GeminiKeyManager(geminiKeysString);
        const geminiClient = new GeminiClient(keyManager);

        // 2. State Management Parsing
        const callerPhone = queryParams.ApiPhone || "Unknown";
        let state = StateManager.decodeState(queryParams.app_state || "");
        let step = state.step || 1;

        const responseBuilder = new YemotResponseBuilder();

        // --------------------------------------------------------------------
        // STEP 1: GREETING & REQUEST AUDIO RECORDING
        // --------------------------------------------------------------------
        if (step === 1) {
            Logger.info(`[Step 1] Asking user ${callerPhone} to record request.`);
            
            // Advance state for next call
            state.step = 2;
            const stateStr = StateManager.encodeState(state);
            
            // Send read command: prompt user to record.
            // Format: read=t-Text=VariableName,Yes/No(ask),min,max,type(record),blockStar
            // Notice we pass the state in the variable name so it comes back to us
            const promptText = "שלום! אני הבינה המלאכותית. אנא הקליטו אחרי הצפצוף איזה קו אתם מחפשים. למשל, אני רוצה לעבור לקו עם הרבה תכנים של מוזיקה. בסיום ההקלטה הקישו סולמית.";
            const varName = `audio_input,no,1,15,record,yes&api_add_0=app_state=${stateStr}`;
            
            return res.send(`read=t-${promptText}=${varName}`);
        }

        // --------------------------------------------------------------------
        // STEP 2: PROCESS AUDIO, SEARCH DB, CREATE MENU & ROUTES
        // --------------------------------------------------------------------
        if (step === 2) {
            Logger.info(`[Step 2] Processing audio from user ${callerPhone}.`);
            
            // Yemot sends the recorded file path in the variable we named 'audio_input'
            const audioFilePath = queryParams.audio_input; 
            
            if (!audioFilePath) {
                return res.send("id_list_message=t-לא זוהתה הקלטה.&go_to_folder=hangup");
            }

            // A. Download Audio
            const audioBuffer = await yemotApi.downloadAudio(audioFilePath);
            
            // B. Send to Gemini for NLP Extraction
            const geminiResult = await geminiClient.safeProcessAudio(audioBuffer);
            
            // C. Clean Keywords using our HebrewNLP
            let cleanKeywords =[];
            geminiResult.keywords.forEach(kw => {
                cleanKeywords.push(...HebrewNLP.tokenize(kw));
            });
            // Add transcribed words just in case Gemini missed something
            cleanKeywords.push(...HebrewNLP.tokenize(geminiResult.transcription));
            // Remove duplicates
            cleanKeywords = [...new Set(cleanKeywords)];

            // D. Local Search inside lines.txt
            const matches = searchEngine.search(cleanKeywords, 5); // Get top 5 max

            if (matches.length === 0) {
                // Return to step 1
                state.step = 1;
                const stateStr = StateManager.encodeState(state);
                return res.send(`id_list_message=t-סליחה, לא מצאתי קווים שמתאימים לבקשה שלכם.&api_add_0=app_state=${stateStr}&go_to_folder=.`);
            }

            // E. Create Dynamic Nitoviya Routes & Build Menu TTS
            // We use a dedicated dynamic folder in Yemot, e.g., /999
            const dynamicBaseFolder = "/999"; 
            
            let menuTts = `מצאתי ${matches.length} קווים שמתאימים לבקשה שלכם. `;
            let validKeys = "";
            let routingOptions = {};

            for (let i = 0; i < matches.length; i++) {
                const match = matches[i];
                const keyPress = i + 1; // 1, 2, 3...
                validKeys += keyPress;
                
                menuTts += `למעבר לקו ${match.name}, ${match.desc}, הקישו ${keyPress}. `;
                
                // Set up Zero-Cost Routing (Nitoviya) dynamically via API
                const extPath = `${dynamicBaseFolder}/route_${callerPhone}_${keyPress}`;
                await yemotApi.setupNitoviyaRouting(extPath, match.phone);
                
                routingOptions[keyPress] = extPath;
            }

            menuTts += "לסיום, הקישו סולמית.";
            
            // Advance state to step 3, store the routing options
            state.step = 3;
            state.routes = routingOptions;
            const stateStr = StateManager.encodeState(state);

            // Send `read` command for digits (User selects 1, 2, 3...)
            // Format: read=t-Text=Var,Ask,Min,Max,Type,Block*,AllowedKeys
            const readParams = `menu_selection,no,1,1,Digits,yes,${validKeys}#&api_add_0=app_state=${stateStr}`;
            return res.send(`read=t-${menuTts}=${readParams}`);
        }

        // --------------------------------------------------------------------
        // STEP 3: EXECUTE DYNAMIC ROUTING (NITOVIYA)
        // --------------------------------------------------------------------
        if (step === 3) {
            Logger.info(`[Step 3] Routing user ${callerPhone}.`);
            
            const selection = queryParams.menu_selection;
            const routes = state.routes;

            if (selection === "#" || !routes[selection]) {
                return res.send("id_list_message=t-היציאה בהצלחה.&go_to_folder=hangup");
            }

            const targetExtension = routes[selection];
            
            // Route the user to the dynamically created Nitoviya extension!
            // No unit costs applied as per Nitoviya specification!
            return res.send(`id_list_message=t-מיד תועברו לקו שבחרתם בהצלחה.&go_to_folder=${targetExtension}`);
        }

    } catch (error) {
        Logger.error("Critical System Error", error);
        res.send("id_list_message=t-אירעה שגיאת מערכת בלתי צפויה. אנא נסו שוב מאוחר יותר.&go_to_folder=hangup");
    }
});

// Start Server (For local testing, Vercel exports the app automatically)
const PORT = process.env.PORT || 3000;
if (process.env.NODE_ENV !== 'production') {
    app.listen(PORT, () => Logger.info(`Server running locally on port ${PORT}`));
}

module.exports = app;
