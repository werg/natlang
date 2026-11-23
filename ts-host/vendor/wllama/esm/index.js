var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __knownSymbol = (name, symbol) => (symbol = Symbol[name]) ? symbol : Symbol.for("Symbol." + name);
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var __await = function(promise, isYieldStar) {
  this[0] = promise;
  this[1] = isYieldStar;
};
var __asyncGenerator = (__this, __arguments, generator) => {
  var resume = (k, v, yes, no) => {
    try {
      var x = generator[k](v), isAwait = (v = x.value) instanceof __await, done = x.done;
      Promise.resolve(isAwait ? v[0] : v).then((y) => isAwait ? resume(k === "return" ? k : "next", v[1] ? { done: y.done, value: y.value } : y, yes, no) : yes({ value: y, done })).catch((e) => resume("throw", e, yes, no));
    } catch (e) {
      no(e);
    }
  }, method = (k) => it[k] = (x) => new Promise((yes, no) => resume(k, x, yes, no)), it = {};
  return generator = generator.apply(__this, __arguments), it[__knownSymbol("asyncIterator")] = () => it, method("next"), method("throw"), method("return"), it;
};
var __forAwait = (obj, it, method) => (it = obj[__knownSymbol("asyncIterator")]) ? it.call(obj) : (obj = obj[__knownSymbol("iterator")](), it = {}, method = (key, fn) => (fn = obj[key]) && (it[key] = (arg) => new Promise((yes, no, done) => (arg = fn.call(obj, arg), done = arg.done, Promise.resolve(arg.value).then((value) => yes({ value, done }), no)))), method("next"), method("return"), it);

// src/glue/messages.ts
var GLUE_VERSION = 1;
var GLUE_MESSAGE_PROTOTYPES = {
  "erro_evt": {
    "name": "erro_evt",
    "structName": "glue_msg_error",
    "className": "GlueMsgError",
    "fields": [
      {
        "type": "str",
        "name": "message",
        "isNullable": false
      }
    ]
  },
  "load_req": {
    "name": "load_req",
    "structName": "glue_msg_load_req",
    "className": "GlueMsgLoadReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "model_paths",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "mmproj_path",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "n_ctx_auto",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mmap",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mlock",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_gpu_layers",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_threads",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "model_alias",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "log_level",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "embeddings",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "offload_kqv",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_parallel",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "pooling_type",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "rope_scaling_type",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_base",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_scale",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_ext_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_attn_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_fast",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_slow",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "yarn_orig_ctx",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_k",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_v",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "kv_unified",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "flash_attn",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "swa_full",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ctx_checkpoints",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "checkpoint_min_step",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "chat_template",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "jinja",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_vals",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "reasoning",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_min_tokens",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_max_tokens",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "warmup",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "no_kv_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "mmproj_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cont_batching",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_keep",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "ctx_shift",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cache_idle_slots",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_cache_reuse",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "lora_paths",
        "isNullable": true
      },
      {
        "type": "arr_float",
        "name": "lora_scales",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "lora_init_without_apply",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "native_state_paths",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "spec_draft_model",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_ngl",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_max",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_min",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "spec_draft_p_min",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads_batch",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_vals",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "reasoning_budget_tokens",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_budget_message",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_format",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "skip_chat_parsing",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "prefill_assistant",
        "isNullable": true
      }
    ]
  },
  "load_res": {
    "name": "load_res",
    "structName": "glue_msg_load_res",
    "className": "GlueMsgLoadRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_vocab",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx_train",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_embd",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_layer",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_key",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_val",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_bos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eot",
        "isNullable": false
      },
      {
        "type": "arr_int",
        "name": "list_tokens_eog",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_bos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_eos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_encoder",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_decoder_start",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "media_marker",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_image_input",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_audio_input",
        "isNullable": false
      }
    ]
  },
  "cmpl_req": {
    "name": "cmpl_req",
    "structName": "glue_msg_completion_req",
    "className": "GlueMsgCompletionReq",
    "fields": [
      {
        "type": "bool",
        "name": "is_chat",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "cmpl_res": {
    "name": "cmpl_res",
    "structName": "glue_msg_completion_res",
    "className": "GlueMsgCompletionRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "embd_req": {
    "name": "embd_req",
    "structName": "glue_msg_embedding_req",
    "className": "GlueMsgEmbeddingReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "embd_res": {
    "name": "embd_res",
    "structName": "glue_msg_embedding_res",
    "className": "GlueMsgEmbeddingRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "rrnk_req": {
    "name": "rrnk_req",
    "structName": "glue_msg_rerank_req",
    "className": "GlueMsgRerankReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "rrnk_res": {
    "name": "rrnk_res",
    "structName": "glue_msg_rerank_res",
    "className": "GlueMsgRerankRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "sys1_req": {
    "name": "sys1_req",
    "structName": "glue_msg_systemone_req",
    "className": "GlueMsgSystemoneReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "sys1_res": {
    "name": "sys1_res",
    "structName": "glue_msg_systemone_res",
    "className": "GlueMsgSystemoneRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "scor_req": {
    "name": "scor_req",
    "structName": "glue_msg_score_req",
    "className": "GlueMsgScoreReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "scor_res": {
    "name": "scor_res",
    "structName": "glue_msg_score_res",
    "className": "GlueMsgScoreRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "slot_req": {
    "name": "slot_req",
    "structName": "glue_msg_slots_req",
    "className": "GlueMsgSlotsReq",
    "fields": [
      {
        "type": "int",
        "name": "id_slot",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "slot_res": {
    "name": "slot_res",
    "structName": "glue_msg_slots_res",
    "className": "GlueMsgSlotsRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "nsta_req": {
    "name": "nsta_req",
    "structName": "glue_msg_native_state_req",
    "className": "GlueMsgNativeStateReq",
    "fields": [
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "id",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "nsta_res": {
    "name": "nsta_res",
    "structName": "glue_msg_native_state_res",
    "className": "GlueMsgNativeStateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "meta_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_req": {
    "name": "tmpl_req",
    "structName": "glue_msg_apply_template_req",
    "className": "GlueMsgApplyTemplateReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_res": {
    "name": "tmpl_res",
    "structName": "glue_msg_apply_template_res",
    "className": "GlueMsgApplyTemplateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "prompt",
        "isNullable": false
      }
    ]
  },
  "gres_req": {
    "name": "gres_req",
    "structName": "glue_msg_get_result_req",
    "className": "GlueMsgGetResultReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_res": {
    "name": "gres_res",
    "structName": "glue_msg_get_result_res",
    "className": "GlueMsgGetResultRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_more",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "is_error",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "cncl_req": {
    "name": "cncl_req",
    "structName": "glue_msg_cancel_req",
    "className": "GlueMsgCancelReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "cncl_res": {
    "name": "cncl_res",
    "structName": "glue_msg_cancel_res",
    "className": "GlueMsgCancelRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  },
  "tbop_req": {
    "name": "tbop_req",
    "structName": "glue_msg_test_backend_ops_req",
    "className": "GlueMsgTestBackendOpsReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "args",
        "isNullable": false
      }
    ]
  },
  "tbop_res": {
    "name": "tbop_res",
    "structName": "glue_msg_test_backend_ops_res",
    "className": "GlueMsgTestBackendOpsRes",
    "fields": [
      {
        "type": "int",
        "name": "retcode",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  }
};

// src/glue/glue.ts
var GLUE_MAGIC = new Uint8Array([71, 76, 85, 69]);
var GLUE_DTYPE_NULL = 0;
var GLUE_DTYPE_BOOL = 1;
var GLUE_DTYPE_INT = 2;
var GLUE_DTYPE_FLOAT = 3;
var GLUE_DTYPE_STRING = 4;
var GLUE_DTYPE_RAW = 5;
var GLUE_DTYPE_ARRAY_BOOL = 6;
var GLUE_DTYPE_ARRAY_INT = 7;
var GLUE_DTYPE_ARRAY_FLOAT = 8;
var GLUE_DTYPE_ARRAY_STRING = 9;
var GLUE_DTYPE_ARRAY_RAW = 10;
var TYPE_MAP = {
  str: GLUE_DTYPE_STRING,
  int: GLUE_DTYPE_INT,
  float: GLUE_DTYPE_FLOAT,
  bool: GLUE_DTYPE_BOOL,
  raw: GLUE_DTYPE_RAW,
  arr_str: GLUE_DTYPE_ARRAY_STRING,
  arr_int: GLUE_DTYPE_ARRAY_INT,
  arr_float: GLUE_DTYPE_ARRAY_FLOAT,
  arr_bool: GLUE_DTYPE_ARRAY_BOOL,
  arr_raw: GLUE_DTYPE_ARRAY_RAW,
  null: GLUE_DTYPE_NULL
};
function glueDeserialize(buf) {
  let offset = 0;
  const view = new DataView(buf.buffer);
  const readUint32 = () => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readInt32 = () => {
    const value = view.getInt32(offset, true);
    offset += 4;
    return value;
  };
  const readFloat = () => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };
  const readBool = () => {
    return readUint32() !== 0;
  };
  const readString = (customLen) => {
    const length = customLen != null ? customLen : readUint32();
    const value = new TextDecoder().decode(buf.slice(offset, offset + length));
    offset += length;
    return value;
  };
  const readRaw = () => {
    const length = readUint32();
    const value = buf.slice(offset, offset + length);
    offset += length;
    return value;
  };
  const readArray = (readItem) => {
    const length = readUint32();
    const value = new Array(length);
    for (let i = 0; i < length; i++) {
      value[i] = readItem();
    }
    return value;
  };
  const readNull = () => null;
  const readField = (field) => {
    switch (field.type) {
      case "str":
        return readString();
      case "int":
        return readInt32();
      case "float":
        return readFloat();
      case "bool":
        return readBool();
      case "raw":
        return readRaw();
      case "arr_str":
        return readArray(readString);
      case "arr_int":
        return readArray(readInt32);
      case "arr_float":
        return readArray(readFloat);
      case "arr_bool":
        return readArray(readBool);
      case "arr_raw":
        return readArray(readRaw);
      case "null":
        return readNull();
    }
  };
  const magicValid = buf[0] === GLUE_MAGIC[0] && buf[1] === GLUE_MAGIC[1] && buf[2] === GLUE_MAGIC[2] && buf[3] === GLUE_MAGIC[3];
  offset += 4;
  if (!magicValid) {
    throw new Error("Invalid magic number");
  }
  const version = readUint32();
  if (version !== GLUE_VERSION) {
    throw new Error("Invalid version number");
  }
  const name = readString(8);
  const msgProto = GLUE_MESSAGE_PROTOTYPES[name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${name}`);
  }
  const output = { _name: name };
  for (const field of msgProto.fields) {
    const readType = readUint32();
    if (readType === GLUE_DTYPE_NULL) {
      if (!field.isNullable) {
        throw new Error(
          `${name}: Expect field ${field.name} to be non-nullable`
        );
      }
      output[field.name] = null;
      continue;
    }
    if (readType !== TYPE_MAP[field.type]) {
      throw new Error(
        `${name}: Expect field ${field.name} to have type ${field.type}`
      );
    }
    output[field.name] = readField(field);
  }
  return output;
}
function glueSerialize(msg) {
  const msgProto = GLUE_MESSAGE_PROTOTYPES[msg._name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${msg._name}`);
  }
  const bufs = [];
  const writeUint32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeInt32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setInt32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeFloat = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeBool = (value) => {
    writeUint32(value ? 1 : 0);
  };
  const writeString = (value) => {
    const utf8 = new TextEncoder().encode(value);
    writeUint32(utf8.byteLength);
    bufs.push(utf8);
  };
  const writeRaw = (value) => {
    writeUint32(value.byteLength);
    bufs.push(value);
  };
  const writeArray = (value, writeItem) => {
    writeUint32(value.length);
    for (const item of value) {
      writeItem(item);
    }
  };
  const writeNull = () => {
  };
  bufs.push(GLUE_MAGIC);
  writeUint32(GLUE_VERSION);
  {
    const utf8 = new TextEncoder().encode(msg._name);
    bufs.push(utf8);
  }
  for (const field of msgProto.fields) {
    const val = msg[field.name];
    if (!field.isNullable && (val === null || val === void 0)) {
      throw new Error(
        `${msg._name}: Expect field ${field.name} to be non-nullable`
      );
    }
    if (val === null || val === void 0) {
      writeUint32(GLUE_DTYPE_NULL);
      continue;
    }
    writeUint32(TYPE_MAP[field.type]);
    switch (field.type) {
      case "str":
        writeString(val);
        break;
      case "int":
        writeInt32(val);
        break;
      case "float":
        writeFloat(val);
        break;
      case "bool":
        writeBool(val);
        break;
      case "raw":
        writeRaw(val);
        break;
      case "arr_str":
        writeArray(val, writeString);
        break;
      case "arr_int":
        writeArray(val, writeInt32);
        break;
      case "arr_float":
        writeArray(val, writeFloat);
        break;
      case "arr_bool":
        writeArray(val, writeBool);
        break;
      case "arr_raw":
        writeArray(val, writeRaw);
        break;
      case "null":
        writeNull();
        break;
    }
  }
  const totalLength = bufs.reduce((acc, buf) => acc + buf.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const buf of bufs) {
    output.set(buf, offset);
    offset += buf.byteLength;
  }
  return output;
}

// src/wasm/source-map.ts
var WASM_SOURCE_MAP = {
  "default": "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",
  "compat": "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"
};

// src/debug.ts
var cache = /* @__PURE__ */ new Map();
function loadMap(buildKey) {
  return __async(this, null, function* () {
    if (cache.has(buildKey)) return cache.get(buildKey);
    const b64 = WASM_SOURCE_MAP[buildKey];
    if (!b64) throw new Error(`No source map for build "${buildKey}"`);
    const gzipped = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const ds = new DecompressionStream("gzip");
    const writer = ds.writable.getWriter();
    writer.write(gzipped);
    writer.close();
    const buf = yield new Response(ds.readable).arrayBuffer();
    const dv = new DataView(buf);
    const bytes = new Uint8Array(buf);
    const firstId = dv.getUint32(0, true);
    const funcCount = dv.getUint32(4, true);
    const numNames = dv.getUint32(8, true);
    const td = new TextDecoder();
    const names = [];
    let pos = 12;
    for (let i = 0; i < numNames; i++) {
      const len = bytes[pos++];
      names.push(td.decode(bytes.subarray(pos, pos + len)));
      pos += len;
    }
    const funcNames = [];
    for (let i = 0; i < funcCount; i++) {
      const idx = dv.getUint16(pos, true);
      pos += 2;
      funcNames.push(idx === 65535 ? null : names[idx]);
    }
    const entry = { firstId, funcNames };
    cache.set(buildKey, entry);
    return entry;
  });
}
var Debug = {
  /**
   * Resolves a list of wasm function indices to their cleaned symbol names.
   */
  decodeFuncIds: (funcIds, isCompatBuild) => __async(void 0, null, function* () {
    const buildKey = isCompatBuild ? "compat" : "default";
    const { firstId, funcNames } = yield loadMap(buildKey);
    return funcIds.map((funcId) => {
      const i = funcId - firstId;
      const name = i >= 0 && i < funcNames.length && funcNames[i] ? funcNames[i] : "(unknown)";
      return { funcId, name };
    });
  }),
  /**
   * Annotates a wasm stack trace string with resolved function names.
   *
   * Example input from Chrome:
   *   at http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[775]:0x74251
   *   at async blob:http://localhost:8080/53a863cc-7227-45cc-8594-ddbbf5257f20:317:28
   *
   * Example input from Firefox:
   *   @http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[796]:0x7dfe2
   *       at wModuleInit/WebAssembly.promising/< (9b6a2acd-d909-44e2-b021-d42fb9087cfb:15:32) index.js:1433:45
   *
   * Example input from Safari:
   *   2441@wasm-function[2441]
   *       at wrapper (d746f19e-4523-4f36-ba06-d0969acc0b05:22:126009)
   *
   * Example output:
   *   wasm-func[775] (server_response::send)
   */
  decodeStackTrace: (stack, isCompatBuild) => __async(void 0, null, function* () {
    const re = /wasm-function\[(\d+)\]/g;
    const funcIds = [
      ...new Set([...stack.matchAll(re)].map((m) => parseInt(m[1])))
    ];
    if (funcIds.length === 0) return stack;
    const resolved = yield Debug.decodeFuncIds(funcIds, isCompatBuild);
    return resolved.map((r) => {
      if (r.name === "(unknown)") {
        return `    wasm-func[${r.funcId}] (unknown)`;
      }
      return `    wasm-func[${r.funcId}] (${r.name})`;
    }).join("\n");
  })
};

// src/utils.ts
var textDecoder = new TextDecoder();
var URL_PARTS_REGEX = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
var parseShardNumber = (fnameOrUrl) => {
  const matches = fnameOrUrl.match(URL_PARTS_REGEX);
  if (!matches) {
    return {
      baseURL: fnameOrUrl,
      current: 1,
      total: 1
    };
  } else {
    return {
      baseURL: fnameOrUrl.replace(URL_PARTS_REGEX, ""),
      current: parseInt(matches[1]),
      total: parseInt(matches[2])
    };
  }
};
var sortFileByShard = (blobs) => {
  const isFiles = blobs.every((b) => !!b.name);
  if (isFiles && blobs.length > 1) {
    const files = blobs;
    files.sort((a, b) => {
      const infoA = parseShardNumber(a.name);
      const infoB = parseShardNumber(b.name);
      return infoA.current - infoB.current;
    });
  }
};
var isMmproj = (blob) => __async(void 0, null, function* () {
  const META_NAME = "general.architecture";
  const META_VAL = "clip";
  const tmp = blob.slice(0, 128 * 1024);
  const header = yield tmp.arrayBuffer();
  const buf = new Uint8Array(header);
  const nameBytes = new TextEncoder().encode(META_NAME);
  const valBytes = new TextEncoder().encode(META_VAL);
  let offset = -1;
  outer: for (let i = 0; i <= buf.length - nameBytes.length; i++) {
    for (let j = 0; j < nameBytes.length; j++) {
      if (buf[i + j] !== nameBytes[j]) continue outer;
    }
    offset = i;
    break;
  }
  if (offset === -1) return false;
  if (offset + 8 * 4 + 4 > buf.length) return false;
  const view = new DataView(header);
  const valLen = view.getBigUint64(offset + 8 * 3, true);
  if (valLen !== /* @__PURE__ */ BigInt("4")) return false;
  for (let i = 0; i < valBytes.length; i++) {
    if (buf[offset + 8 * 4 + i] !== valBytes[i]) return false;
  }
  return true;
});
var absoluteUrl = (relativePath) => typeof document === "undefined" ? new URL(relativePath, self.location.href).href : new URL(relativePath, document.baseURI).href;
var padDigits = (number, digits) => {
  return Array(Math.max(digits - String(number).length + 1, 0)).join("0") + number;
};
var sumArr = (arr) => arr.reduce((prev, curr) => prev + curr, 0);
var isString = (value) => !!(value == null ? void 0 : value.startsWith);
var MMPROJ_FILE_NAME = "mmproj.gguf";
var prepareBlobs = (blobsInp) => __async(void 0, null, function* () {
  const blobs = [];
  let blobMmproj = null;
  for (const blob of blobsInp) {
    if (yield isMmproj(blob)) {
      blobMmproj = blob;
    } else {
      blobs.push(blob);
    }
  }
  sortFileByShard(blobs);
  const result = blobs.map((blob, i) => ({
    blob,
    name: `model-${padDigits(i + 1, 5)}-of-${padDigits(blobs.length, 5)}.gguf`
  }));
  if (blobMmproj) {
    result.push({
      blob: blobMmproj,
      name: MMPROJ_FILE_NAME
    });
  }
  return {
    llm: result.filter((f) => f.name !== MMPROJ_FILE_NAME),
    mmproj: blobMmproj ? { blob: blobMmproj, name: MMPROJ_FILE_NAME } : null,
    all: result
  };
});
var isSupportMultiThread = () => ((e) => __async(void 0, null, function* () {
  try {
    return "undefined" != typeof MessageChannel && new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)), WebAssembly.validate(e);
  } catch (e2) {
    return false;
  }
}))(
  new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    1,
    4,
    1,
    96,
    0,
    0,
    3,
    2,
    1,
    0,
    5,
    4,
    1,
    3,
    1,
    1,
    10,
    11,
    1,
    9,
    0,
    65,
    0,
    254,
    16,
    2,
    0,
    26,
    11
  ])
);
var isSupportExceptions = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      4,
      1,
      96,
      0,
      0,
      3,
      2,
      1,
      0,
      10,
      8,
      1,
      6,
      0,
      6,
      64,
      25,
      11,
      11
    ])
  );
});
var isSupportSIMD = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      5,
      1,
      96,
      0,
      1,
      123,
      3,
      2,
      1,
      0,
      10,
      10,
      1,
      8,
      0,
      65,
      0,
      253,
      15,
      253,
      98,
      11
    ])
  );
});
var isSupportJSPI = () => {
  return !!WebAssembly.Suspending;
};
var isSupportWebGPU = () => {
  return !!navigator.gpu;
};
var isSupportMem64 = () => {
  try {
    new WebAssembly.Memory({
      address: "i64",
      initial: /* @__PURE__ */ BigInt("1")
      // 1 page (64 KiB)
    });
    return true;
  } catch (e) {
    return false;
  }
};
var checkEnvironmentCompatible = () => __async(void 0, null, function* () {
  if (!(yield isSupportExceptions())) {
    throw new Error("WebAssembly runtime does not support exception handling");
  }
  if (!(yield isSupportSIMD())) {
    throw new Error("WebAssembly runtime does not support SIMD");
  }
});
var isFirefox = () => {
  return !!navigator.userAgent.match(/Firefox\/([0-9\.]+)(?:\s|$)/);
};
var GGUF_FILE_REGEX = /^.*\.gguf(?:\?.*)?$/;
var isValidGgufFile = (path) => {
  return GGUF_FILE_REGEX.test(path);
};
var isSafariMobile = () => {
  return !!navigator.userAgent.match(/Version\/([0-9\._]+).*Mobile.*Safari.*/);
};
var createWorker = (workerCode) => {
  const workerURL = URL.createObjectURL(
    isString(workerCode) ? new Blob([workerCode], { type: "text/javascript" }) : workerCode
  );
  return new Worker(workerURL, { type: "module" });
};
var cbToAsyncIter = (fn) => (...args) => {
  let values = [];
  let resolve;
  let reject;
  values.push(
    new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    })
  );
  fn(...args, (val, done, err) => {
    if (err) {
      reject(err);
      return;
    }
    resolve([val, done]);
    values.push(
      new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      })
    );
  });
  return function() {
    return __asyncGenerator(this, null, function* () {
      let val;
      for (let i = 0, done = false; !done; i++) {
        [val, done] = yield new __await(values[i]);
        delete values[i];
        if (val !== void 0) yield val;
      }
    });
  }();
};
var canUseAsyncFileRead = (compat) => isSupportJSPI() || compat;
var needCompat = () => !isSupportJSPI() || !isSupportMem64();

// src/workers-code/generated.ts
var LIBLLAMA_VERSION = "b12092-1a7f501";
var LLAMA_CPP_WORKER_CODE = "// Start the main llama.cpp\nlet wllamaMalloc;\nlet wllamaStart;\nlet wllamaAction;\nlet wllamaExit;\nlet wllamaDebug;\n\nlet Module = null;\nlet isCompat = false;\nlet lastStack = '';\nlet isAborted = false;\nlet hasMultithread = false;\n\n//////////////////////////////////////////////////////////////\n// UTILS\n//////////////////////////////////////////////////////////////\n\n// send message back to main thread\nconst msg = (data, transfer) => postMessage(data, transfer);\n\n// Convert CPP log into JS log\nconst cppLogToJSLog = (line) => {\n  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);\n  return !!matched\n    ? {\n        level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),\n        text: matched[2],\n      }\n    : { level: 'log', text: line };\n};\n\nconst getHeapU8 = () => {\n  const buffer = Module.wasmMemory.buffer;\n  return new Uint8Array(buffer);\n};\n\nconst toSizeT = (num) => {\n  return isCompat ? Number(num) : BigInt(num);\n};\n\n// Get module config that forwards stdout/err to main thread\nconst getWModuleConfig = (_argMainScriptBlob) => {\n  var pathConfig = RUN_OPTIONS.pathConfig;\n  var pthreadPoolSize = RUN_OPTIONS.nbThread;\n  var argMainScriptBlob = _argMainScriptBlob;\n\n  isCompat = RUN_OPTIONS.compat;\n  hasMultithread = pthreadPoolSize > 1;\n\n  msg({\n    verb: 'console.debug',\n    args: [\n      `Multithread enabled: ${hasMultithread}, pthreadPoolSize: ${pthreadPoolSize}`,\n    ],\n  });\n\n  if (!pathConfig['wllama.wasm']) {\n    throw new Error('\"wllama.wasm\" is missing in pathConfig');\n  }\n  return {\n    noInitialRun: true,\n    print: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      msg({ verb: 'console.log', args: [text] });\n    },\n    printErr: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      if (text.startsWith('@@STACK@@')) {\n        lastStack = text.slice('@@STACK@@'.length);\n        return;\n      }\n      const logLine = cppLogToJSLog(text);\n      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });\n    },\n    locateFile: function (filename, basePath) {\n      const p = pathConfig[filename];\n      const truncate = (str) =>\n        str.length > 128 ? `${str.substr(0, 128)}...` : str;\n      if (filename.match(/wllama\\.worker\\.js/)) {\n        msg({\n          verb: 'console.error',\n          args: [\n            '\"wllama.worker.js\" is removed from v2.2.1. Hint: make sure to clear browser\\'s cache.',\n          ],\n        });\n      } else {\n        msg({\n          verb: 'console.debug',\n          args: [`Loading \"${filename}\" from \"${truncate(p)}\"`],\n        });\n        return p;\n      }\n    },\n    mainScriptUrlOrBlob: hasMultithread\n      ? argMainScriptBlob\n      : 'throw new Error(\"Multithreading is not enabled\")',\n    pthreadPoolSize: hasMultithread ? pthreadPoolSize : 0,\n    wasmMemory: hasMultithread ? getWasmMemory() : null,\n    onAbort: function (message) {\n      isAborted = true;\n      msg({ verb: 'signal.abort', args: ['abort', message, lastStack, null] });\n    },\n    onExit: function (code) {\n      isAborted = true;\n      const callstack = new Error().stack.toString();\n      msg({\n        verb: 'signal.abort',\n        args: ['abort', 'exit(' + code + ')', callstack, null],\n      });\n    },\n  };\n};\n\n// Get the memory to be used by wasm. (Only used in multi-thread mode)\n// Because we have a weird OOM issue on iOS, we need to try some values\n// See: https://github.com/emscripten-core/emscripten/issues/19144\n//      https://github.com/godotengine/godot/issues/70621\nconst getWasmMemory = () => {\n  let minBytes = 128 * 1024 * 1024;\n  let maxBytes = 4096 * 1024 * 1024;\n  let stepBytes = 128 * 1024 * 1024;\n  while (maxBytes > minBytes) {\n    try {\n      const wasmMemory = new WebAssembly.Memory({\n        initial: toSizeT(minBytes / 65536),\n        maximum: toSizeT(maxBytes / 65536),\n        shared: true,\n        address: isCompat ? undefined : 'i64',\n      });\n      return wasmMemory;\n    } catch (e) {\n      maxBytes -= stepBytes;\n      continue; // retry\n    }\n  }\n  throw new Error('Cannot allocate WebAssembly.Memory');\n};\n\n//////////////////////////////////////////////////////////////\n// HEAPFS PATCH\n//////////////////////////////////////////////////////////////\n\n/**\n * By default, emscripten uses memfs. The way it works is by\n * allocating new Uint8Array in javascript heap. This is not good\n * because it requires files to be copied to wasm heap each time\n * a file is read.\n *\n * HeapFS is an alternative, which resolves this problem by\n * allocating space for file directly inside wasm heap. This\n * allows us to mmap without doing any copy.\n *\n * For llama.cpp, this is great because we use MAP_SHARED\n *\n * Ref: https://github.com/ngxson/wllama/pull/39\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js\n *\n * Note 29/05/2024 @ngxson\n * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c\n */\n\nconst fsNameToFile = {}; // map Name => File\nconst fsIdToFile = {}; // map ID => File\nlet currFileId = 0;\n\n// Patch and redirect memfs calls to wllama\nconst patchHeapFS = () => {\n  const m = Module;\n  // save functions\n  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;\n  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;\n  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;\n  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;\n  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;\n  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;\n\n  const patchStream = (stream) => {\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const ptr = Number(f.ptr);\n      stream.node.contents = getHeapU8().subarray(ptr, ptr + f.size);\n      stream.node.usedBytes = f.size;\n    }\n  };\n\n  // replace \"read\" functions\n  m.MEMFS.stream_ops.read = function (\n    stream,\n    buffer,\n    offset,\n    length,\n    position\n  ) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);\n  };\n  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;\n\n  // replace \"llseek\" functions\n  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._llseek(stream, offset, whence);\n  };\n  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;\n\n  // replace \"mmap\" functions\n  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {\n    patchStream(stream);\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const mmapPtr = f.ptr + toSizeT(position);\n      return {\n        ptr: mmapPtr,\n        allocated: false,\n      };\n    } else {\n      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);\n    }\n  };\n  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;\n\n  // mount FS\n  m.FS.mkdir('/models');\n  m.FS.mount(m.MEMFS, { root: '.' }, '/models');\n  // slot save files (action \"slots\"), moved to and from the main thread as bytes\n  m.FS.mkdir('/slots');\n};\n\n// Allocate a new file in wllama heapfs, returns file ID\nconst heapfsAlloc = (name, size, allocBuffer) => {\n  if (size < 1) {\n    throw new Error('File size must be bigger than 0');\n  }\n  const m = Module;\n  const ptr = toSizeT(allocBuffer ? m.mmapAlloc(size) : 0);\n  const file = {\n    ptr: ptr,\n    size: size,\n    id: currFileId++,\n  };\n  fsIdToFile[file.id] = file;\n  fsNameToFile[name] = file;\n  return file.id;\n};\n\n// Add new file to wllama heapfs, return number of written bytes\nconst heapfsWrite = (id, buffer, offset) => {\n  if (fsIdToFile[id]) {\n    const { ptr, size } = fsIdToFile[id];\n    const afterWriteByte = offset + buffer.byteLength;\n    if (afterWriteByte > size) {\n      throw new Error(\n        `File ID ${id} write out of bound, afterWriteByte = ${afterWriteByte} while size = ${size}`\n      );\n    }\n    getHeapU8().set(buffer, Number(ptr) + offset);\n    return buffer.byteLength;\n  } else {\n    throw new Error(`File ID ${id} not found in heapfs`);\n  }\n};\n\n//////////////////////////////////////////////////////////////\n// ASYNC FILE READ\n//////////////////////////////////////////////////////////////\n\nlet isAwaitReading = false;\nlet pendingReadPromise = null;\nlet pendingReadResolve = null;\nlet pendingReadReject = null;\n\nconst _stripModelsPrefix = (path) => path.replace(/^\\/?models\\//, '');\n\n// Called from EM_ASYNC_JS stub in wllama-fs.h (path is already a JS string)\nconst _wllama_js_file_read = async (path, offset, req_size, out_ptr) => {\n  const name = _stripModelsPrefix(path);\n\n  pendingReadPromise = new Promise((res, rej) => {\n    pendingReadResolve = res;\n    pendingReadReject = rej;\n  });\n  isAwaitReading = true;\n\n  postMessage({ verb: 'fs.read_req', args: [name, offset, req_size] });\n\n  let data;\n  try {\n    data = await pendingReadPromise;\n  } finally {\n    isAwaitReading = false;\n    pendingReadResolve = null;\n    pendingReadReject = null;\n  }\n\n  const bytes = new Uint8Array(data);\n  getHeapU8().set(bytes, out_ptr);\n  return toSizeT(bytes.length);\n};\n\n//////////////////////////////////////////////////////////////\n// MAIN CODE\n//////////////////////////////////////////////////////////////\n\nconst callWrapper = (name, ret, args, isAsync) => {\n  const fn = Module.cwrap(\n    name,\n    ret,\n    args,\n    isAsync ? { async: true } : undefined\n  );\n  return async (action, req) => {\n    // console.log(`Calling ${name} with action:`, action, 'and req:', req);\n    let result;\n    try {\n      if (args.length === 2) {\n        result = isAsync ? await fn(action, req) : fn(action, req);\n      } else {\n        result = fn();\n      }\n    } catch (ex) {\n      console.error(ex);\n      throw ex;\n    }\n    return result;\n  };\n};\n\n// re-entering the wasm while a call is suspended (JSPI / asyncify) corrupts its state, so only one call runs at a time and the rest wait in the queue\nlet wasmCallBusy = false;\nconst wasmCallQueue = [];\n\nconst runWasmCall = async (callbackId, fn) => {\n  if (isAborted) {\n    // the wasm is dead, fail fast instead of calling into it\n    msg({ callbackId, err: 'wllama has crashed, please reload the module' });\n    return;\n  }\n  if (wasmCallBusy) {\n    wasmCallQueue.push({ callbackId, fn });\n    return;\n  }\n  wasmCallBusy = true;\n  try {\n    await fn();\n  } finally {\n    wasmCallBusy = false;\n    if (isAborted) {\n      // do not touch the wasm again after it aborted; the main thread already rejected the queued tasks\n      wasmCallQueue.length = 0;\n    } else {\n      const next = wasmCallQueue.shift();\n      if (next) runWasmCall(next.callbackId, next.fn);\n    }\n  }\n};\n\nconst runAction = async (data) => {\n  const { args, callbackId } = data;\n  const argAction = args[0];\n  const argEncodedMsg = args[1];\n  try {\n    const inputPtr = await wllamaMalloc(toSizeT(argEncodedMsg.byteLength), 0);\n    // copy data to wasm heap\n    const inputBuffer = new Uint8Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      argEncodedMsg.byteLength\n    );\n    inputBuffer.set(argEncodedMsg, 0);\n    const outputPtr = await wllamaAction(argAction, inputPtr);\n    // length of output buffer is written at the first 4 bytes of input buffer\n    const outputLen = new Uint32Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      1\n    )[0];\n    // copy the output buffer to JS heap\n    const outputBuffer = new Uint8Array(outputLen);\n    const outputSrcView = new Uint8Array(\n      getHeapU8().buffer,\n      Number(outputPtr),\n      outputLen\n    );\n    outputBuffer.set(outputSrcView, 0); // copy it\n    msg({ callbackId, result: outputBuffer }, [outputBuffer.buffer]);\n  } catch (err) {\n    handleError(err);\n  }\n};\n\nfunction handleError(err) {\n  // If WASM already aborted, onAbort already sent signal.abort; skip to avoid\n  // re-reporting the resulting WebAssembly.RuntimeError as a JS exception.\n  if (isAborted) return;\n\n  const message = err ? err.message || String(err) : 'Unknown error';\n  const stack = err ? err.stack || String(err) : '';\n  msg({\n    verb: 'signal.abort',\n    args: ['exception', message, stack, err],\n  });\n}\n\nonmessage = async (e) => {\n  if (!e.data) return;\n  const { verb, args, callbackId } = e.data;\n\n  // fs.read_res arrives while wasm is JSPI-suspended; resolve the pending promise.\n  if (verb === 'fs.read_res') {\n    if (pendingReadResolve) {\n      pendingReadResolve(args[0]);\n    }\n    return;\n  }\n\n  // Guard: while awaiting a file read, reject any other incoming task.\n  if (isAwaitReading) {\n    if (callbackId) {\n      msg({\n        callbackId,\n        err: 'Worker is suspended waiting for file data (JSPI)',\n      });\n    }\n    return;\n  }\n\n  if (!callbackId) {\n    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });\n    return;\n  }\n\n  if (verb === 'module.init') {\n    const argMainScriptBlob = args[0];\n    const argUseAsyncFile = args[1];\n    try {\n      Module = getWModuleConfig(argMainScriptBlob);\n      Module.preRun = () => {\n        if (argUseAsyncFile) {\n          Module.ENV['USE_ASYNC_FILE'] = '1';\n        }\n      };\n      Module.onRuntimeInitialized = () => {\n        // async call once module is ready\n        // init FS\n        patchHeapFS();\n        // init cwrap\n        const pointer = isCompat ? 'number' : 'bigint';\n        // TODO: note sure why emscripten cannot bind if there is only 1 argument\n        wllamaMalloc = callWrapper('wllama_malloc', pointer, [\n          'number',\n          pointer,\n        ]);\n        wllamaStart = callWrapper('wllama_start', 'string', [], true);\n        wllamaAction = callWrapper(\n          'wllama_action',\n          pointer,\n          ['string', pointer],\n          true\n        );\n        wllamaExit = callWrapper('wllama_exit', 'string', []);\n        wllamaDebug = callWrapper('wllama_debug', 'string', []);\n        msg({ callbackId, result: null });\n      };\n      wModuleInit();\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.alloc') {\n    const argFilename = args[0];\n    const argSize = args[1];\n    const argAllocBuffer = args[2];\n    try {\n      // create blank file\n      const emptyBuffer = new ArrayBuffer(0);\n      Module['FS_createDataFile'](\n        '/models',\n        argFilename,\n        emptyBuffer,\n        true,\n        true,\n        true\n      );\n      // alloc data on heap\n      const fileId = heapfsAlloc(argFilename, argSize, argAllocBuffer);\n      msg({ callbackId, result: { fileId } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.write') {\n    const argFileId = args[0];\n    const argBuffer = args[1];\n    const argOffset = args[2];\n    try {\n      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);\n      msg({ callbackId, result: { writtenBytes } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_read') {\n    const argFilename = args[0];\n    try {\n      const path = '/slots/' + argFilename;\n      const data = Module.FS.readFile(path);\n      Module.FS.unlink(path);\n      msg({ callbackId, result: data }, [data.buffer]);\n    } catch (err) {\n      msg({ callbackId, err: `cannot read slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_write') {\n    const argFilename = args[0];\n    const argData = args[1];\n    try {\n      Module.FS.writeFile('/slots/' + argFilename, argData);\n      msg({ callbackId, result: null });\n    } catch (err) {\n      msg({ callbackId, err: `cannot write slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_remove') {\n    try {\n      Module.FS.unlink('/slots/' + args[0]);\n    } catch (err) {\n      // already removed\n    }\n    msg({ callbackId, result: null });\n    return;\n  }\n\n  if (verb === 'wllama.start') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaStart();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.action') {\n    await runWasmCall(callbackId, () => runAction(e.data));\n    return;\n  }\n\n  if (verb === 'wllama.exit') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaExit();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.debug') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaDebug();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n};\n";
var OPFS_UTILS_WORKER_CODE = "let accessHandle;\nlet abortController = new AbortController();\n\nasync function openFile(filename) {\n  const opfsRoot = await navigator.storage.getDirectory();\n  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });\n  const fileHandler = await cacheDir.getFileHandle(filename, { create: true });\n  accessHandle = await fileHandler.createSyncAccessHandle();\n  accessHandle.truncate(0); // clear file content\n}\n\nasync function writeFile(buf) {\n  accessHandle.write(buf);\n}\n\nasync function closeFile() {\n  accessHandle.flush();\n  accessHandle.close();\n}\n\nasync function writeTextFile(filename, str) {\n  await openFile(filename);\n  await writeFile(new TextEncoder().encode(str));\n  await closeFile();\n}\n\nconst throttled = (func, delay) => {\n  let lastRun = 0;\n  return (...args) => {\n    const now = Date.now();\n    if (now - lastRun > delay) {\n      lastRun = now;\n      func.apply(null, args);\n    }\n  };\n};\n\nconst assertNonNull = (val) => {\n  if (val === null || val === undefined) {\n    throw new Error('OPFS Worker: Assertion failed');\n  }\n};\n\n// respond to main thread\nconst resOK = () => postMessage({ ok: true });\nconst resProgress = (loaded, total) =>\n  postMessage({ progress: { loaded, total } });\nconst resErr = (err) => postMessage({ err });\n\nonmessage = async (e) => {\n  try {\n    if (!e.data) return;\n\n    /**\n     * @param {Object} e.data\n     *\n     * Fine-control FS actions:\n     * - { action: 'open', filename: 'string' }\n     * - { action: 'write', buf: ArrayBuffer }\n     * - { action: 'close' }\n     *\n     * Simple write API:\n     * - { action: 'write-simple', filename: 'string', buf: ArrayBuffer }\n     *\n     * Download API:\n     * - { action: 'download', url: 'string', filename: 'string', options: Object, metadataFileName: 'string' }\n     * - { action: 'download-abort' }\n     */\n    const {\n      action,\n      filename,\n      buf,\n      url,\n      options,\n      metadataFileName,\n      metadataAdditional,\n    } = e.data;\n\n    if (action === 'open') {\n      assertNonNull(filename);\n      await openFile(filename);\n      return resOK();\n    } else if (action === 'write') {\n      assertNonNull(buf);\n      await writeFile(buf);\n      return resOK();\n    } else if (action === 'close') {\n      await closeFile();\n      return resOK();\n    } else if (action === 'write-simple') {\n      assertNonNull(filename);\n      assertNonNull(buf);\n      await openFile(filename);\n      await writeFile(buf);\n      await closeFile();\n      return resOK();\n    } else if (action === 'download') {\n      assertNonNull(url);\n      assertNonNull(filename);\n      assertNonNull(metadataFileName);\n      assertNonNull(options);\n      assertNonNull(options.aborted);\n      abortController = new AbortController();\n      if (options.aborted) abortController.abort();\n      const response = await fetch(url, {\n        ...options,\n        signal: abortController.signal,\n      });\n      const contentLength = response.headers.get('content-length');\n      const etag = (response.headers.get('etag') || '').replace(\n        /[^A-Za-z0-9]/g,\n        ''\n      );\n      const total = parseInt(contentLength, 10);\n      const reader = response.body.getReader();\n      await openFile(filename);\n      let loaded = 0;\n      const throttledProgress = throttled(resProgress, 100);\n      while (true) {\n        const { done, value } = await reader.read();\n        if (done) break;\n        loaded += value.byteLength;\n        await writeFile(value);\n        throttledProgress(loaded, total);\n      }\n      resProgress(total, total); // 100% done\n      await closeFile();\n      // make sure this is in-sync with CacheEntryMetadata\n      await writeTextFile(\n        metadataFileName,\n        JSON.stringify({\n          originalURL: url,\n          originalSize: total,\n          etag,\n          ...metadataAdditional,\n        })\n      );\n      return resOK();\n    } else if (action === 'download-abort') {\n      if (abortController) {\n        abortController.abort();\n      }\n      return;\n    }\n\n    throw new Error('OPFS Worker: Invalid action', e.data);\n  } catch (err) {\n    return resErr(err);\n  }\n};\n";
var WLLAMA_EMSCRIPTEN_CODE = 'var Module=typeof Module!="undefined"?Module:{};var ENVIRONMENT_IS_WEB=!!globalThis.window;var ENVIRONMENT_IS_WORKER=!!globalThis.WorkerGlobalScope;var ENVIRONMENT_IS_NODE=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";var ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&self.name?.startsWith("em-pthread");if(ENVIRONMENT_IS_NODE){var worker_threads=require("worker_threads");global.Worker=worker_threads.Worker;ENVIRONMENT_IS_WORKER=!worker_threads.isMainThread;ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&worker_threads["workerData"]=="em-pthread"}var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var _scriptName=globalThis.document?.currentScript?.src;if(typeof __filename!="undefined"){_scriptName=__filename}else if(ENVIRONMENT_IS_WORKER){_scriptName=self.location.href}var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("fs");scriptDirectory=__dirname+"/";readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename);return ret};readAsync=async(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename,binary?undefined:"utf8");return ret};if(process.argv.length>1){thisProgram=process.argv[1].replace(/\\\\/g,"/")}arguments_=process.argv.slice(2);if(typeof module!="undefined"){module["exports"]=Module}quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){try{scriptDirectory=new URL(".",_scriptName).href}catch{}if(!ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=async url=>{if(isFileURI(url)){return new Promise((resolve,reject)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){resolve(xhr.response);return}reject(xhr.status)};xhr.onerror=reject;xhr.send(null)})}var response=await fetch(url,{credentials:"same-origin"});if(response.ok){return response.arrayBuffer()}throw new Error(response.status+" : "+response.url)}}}else{}var defaultPrint=console.log.bind(console);var defaultPrintErr=console.error.bind(console);if(ENVIRONMENT_IS_NODE){var utils=require("util");var stringify=a=>typeof a=="object"?utils.inspect(a):a;defaultPrint=(...args)=>fs.writeSync(1,args.map(stringify).join(" ")+"\\n");defaultPrintErr=(...args)=>fs.writeSync(2,args.map(stringify).join(" ")+"\\n")}var out=defaultPrint;var err=defaultPrintErr;var wasmBinary;var wasmModule;var ABORT=false;var EXITSTATUS;function assert(condition,text){if(!condition){abort(text)}}var isFileURI=filename=>filename.startsWith("file://");function growMemViews(){if(wasmMemory.buffer!=HEAP8.buffer){updateMemoryViews()}}if(ENVIRONMENT_IS_NODE&&ENVIRONMENT_IS_PTHREAD){var parentPort=worker_threads["parentPort"];parentPort.on("message",msg=>global.onmessage?.({data:msg}));Object.assign(globalThis,{self:global,postMessage:msg=>parentPort["postMessage"](msg)});process.on("uncaughtException",err=>{postMessage({cmd:"uncaughtException",error:err});process.exit(1)})}var startWorker;if(ENVIRONMENT_IS_PTHREAD){var initializedJS=false;self.onunhandledrejection=e=>{throw e.reason||e};async function handleMessage(e){try{var msgData=e["data"];var cmd=msgData.cmd;if(cmd==="load"){let messageQueue=[];self.onmessage=e=>messageQueue.push(e);startWorker=()=>{postMessage({cmd:"loaded"});for(let msg of messageQueue){handleMessage(msg)}self.onmessage=handleMessage};for(const handler of msgData.handlers){if(!Module[handler]||Module[handler].proxy){Module[handler]=(...args)=>{postMessage({cmd:"callHandler",handler,args})};if(handler=="print")out=Module[handler];if(handler=="printErr")err=Module[handler]}}wasmMemory=msgData.wasmMemory;updateMemoryViews();wasmModule=msgData.wasmModule;createWasm();run()}else if(cmd==="run"){establishStackSpace(msgData.pthread_ptr);__emscripten_thread_init(msgData.pthread_ptr,0,0,1,0,0);PThread.threadInitTLS();__emscripten_thread_mailbox_await(msgData.pthread_ptr);if(!initializedJS){initializedJS=true}try{await invokeEntryPoint(msgData.start_routine,msgData.arg)}catch(ex){if(ex!="unwind"){throw ex}}}else if(msgData.target==="setimmediate"){}else if(cmd==="checkMailbox"){if(initializedJS){checkMailbox()}}else if(cmd){err(`worker: received unknown command ${cmd}`);err(msgData)}}catch(ex){__emscripten_thread_crashed();throw ex}}self.onmessage=handleMessage}var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;var HEAP64,HEAPU64;var runtimeInitialized=false;function updateMemoryViews(){var b=wasmMemory.buffer;HEAP8=new Int8Array(b);HEAP16=new Int16Array(b);Module["HEAPU8"]=HEAPU8=new Uint8Array(b);HEAPU16=new Uint16Array(b);HEAP32=new Int32Array(b);HEAPU32=new Uint32Array(b);HEAPF32=new Float32Array(b);HEAPF64=new Float64Array(b);HEAP64=new BigInt64Array(b);HEAPU64=new BigUint64Array(b)}function initMemory(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"]}else{var INITIAL_MEMORY=Module["INITIAL_MEMORY"]||134217728;wasmMemory=new WebAssembly.Memory({initial:BigInt(INITIAL_MEMORY/65536),maximum:65536n,shared:true,address:"i64"})}updateMemoryViews()}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift())}}callRuntimeCallbacks(onPreRuns)}function initRuntime(){runtimeInitialized=true;if(ENVIRONMENT_IS_PTHREAD)return startWorker();if(!Module["noFSInit"]&&!FS.initialized)FS.init();TTY.init();wasmExports["__wasm_call_ctors"]();FS.ignorePermissions=false}function preMain(){}function postRun(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift())}}callRuntimeCallbacks(onPostRuns)}function abort(what){Module["onAbort"]?.(what);what="Aborted("+what+")";err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";if(runtimeInitialized){___trap()}var e=new WebAssembly.RuntimeError(what);throw e}var wasmBinaryFile;function findWasmBinary(){return locateFile("wllama.wasm")}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}async function getWasmBinary(binaryFile){if(!wasmBinary){try{var response=await readAsync(binaryFile);return new Uint8Array(response)}catch{}}return getBinarySync(binaryFile)}async function instantiateArrayBuffer(binaryFile,imports){try{var binary=await getWasmBinary(binaryFile);var instance=await WebAssembly.instantiate(binary,imports);return instance}catch(reason){err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)}}async function instantiateAsync(binary,binaryFile,imports){if(!binary&&!isFileURI(binaryFile)&&!ENVIRONMENT_IS_NODE){try{var response=fetch(binaryFile,{credentials:"same-origin"});var instantiationResult=await WebAssembly.instantiateStreaming(response,imports);return instantiationResult}catch(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation")}}return instantiateArrayBuffer(binaryFile,imports)}function getWasmImports(){assignWasmImports();if(!wasmImports.__instrumented){wasmImports.__instrumented=true;Asyncify.instrumentWasmImports(wasmImports)}var imports={env:wasmImports,wasi_snapshot_preview1:wasmImports};return imports}async function createWasm(){function receiveInstance(instance,module){wasmExports=instance.exports;wasmExports=Asyncify.instrumentWasmExports(wasmExports);wasmExports=applySignatureConversions(wasmExports);registerTLSInit(wasmExports["_emscripten_tls_init"]);assignWasmExports(wasmExports);wasmModule=module;removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){return receiveInstance(result["instance"],result["module"])}var info=getWasmImports();if(Module["instantiateWasm"]){return new Promise((resolve,reject)=>{Module["instantiateWasm"](info,(inst,mod)=>{resolve(receiveInstance(inst,mod))})})}if(ENVIRONMENT_IS_PTHREAD){var instance=new WebAssembly.Instance(wasmModule,getWasmImports());return receiveInstance(instance,wasmModule)}wasmBinaryFile??=findWasmBinary();var result=await instantiateAsync(wasmBinary,wasmBinaryFile,info);var exports=receiveInstantiationResult(result);return exports}class ExitStatus{name="ExitStatus";constructor(status){this.message=`Program terminated with exit(${status})`;this.status=status}}var terminateWorker=worker=>{worker.terminate();worker.onmessage=e=>{}};var cleanupThread=pthread_ptr=>{var worker=PThread.pthreads[pthread_ptr];PThread.returnWorkerToPool(worker)};var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module)}};var onPreRuns=[];var addOnPreRun=cb=>onPreRuns.push(cb);var runDependencies=0;var dependenciesFulfilled=null;var removeRunDependency=id=>{runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}};var addRunDependency=id=>{runDependencies++;Module["monitorRunDependencies"]?.(runDependencies)};var spawnThread=threadParams=>{var worker=PThread.getNewWorker();if(!worker){return 6}PThread.runningWorkers.push(worker);PThread.pthreads[threadParams.pthread_ptr]=worker;worker.pthread_ptr=threadParams.pthread_ptr;var msg={cmd:"run",start_routine:threadParams.startRoutine,arg:threadParams.arg,pthread_ptr:threadParams.pthread_ptr};if(ENVIRONMENT_IS_NODE){worker.unref()}worker.postMessage(msg,threadParams.transferList);return 0};var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var stackSave=()=>_emscripten_stack_get_current();var stackRestore=val=>__emscripten_stack_restore(val);var stackAlloc=sz=>__emscripten_stack_alloc(sz);var proxyToMainThread=(funcIndex,emAsmAddr,sync,...callArgs)=>{var serializedNumCallArgs=callArgs.length*2;var sp=stackSave();var args=stackAlloc(serializedNumCallArgs*8);var b=args/8;for(var i=0;i<callArgs.length;i++){var arg=callArgs[i];if(typeof arg=="bigint"){(growMemViews(),HEAP64)[b+2*i]=1n;(growMemViews(),HEAP64)[b+2*i+1]=arg}else{(growMemViews(),HEAP64)[b+2*i]=0n;(growMemViews(),HEAPF64)[b+2*i+1]=arg}}var rtn=__emscripten_run_js_on_main_thread(funcIndex,emAsmAddr,serializedNumCallArgs,args,sync);stackRestore(sp);return rtn};function _proc_exit(code){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(0,0,1,code);EXITSTATUS=code;if(!keepRuntimeAlive()){PThread.terminateAllThreads();Module["onExit"]?.(code);ABORT=true}quit_(code,new ExitStatus(code))}function exitOnMainThread(returnCode){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(1,0,0,returnCode);_exit(returnCode)}var exitJS=(status,implicit)=>{EXITSTATUS=status;if(ENVIRONMENT_IS_PTHREAD){exitOnMainThread(status);throw"unwind"}_proc_exit(status)};var _exit=exitJS;var PThread={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init(){if(!ENVIRONMENT_IS_PTHREAD){PThread.initMainThread()}},initMainThread(){var pthreadPoolSize=Module["pthreadPoolSize"];while(pthreadPoolSize--){PThread.allocateUnusedWorker()}addOnPreRun(async()=>{var pthreadPoolReady=PThread.loadWasmModuleToAllWorkers();addRunDependency("loading-workers");await pthreadPoolReady;removeRunDependency("loading-workers")})},terminateAllThreads:()=>{for(var worker of PThread.runningWorkers){terminateWorker(worker)}for(var worker of PThread.unusedWorkers){terminateWorker(worker)}PThread.unusedWorkers=[];PThread.runningWorkers=[];PThread.pthreads={}},returnWorkerToPool:worker=>{var pthread_ptr=worker.pthread_ptr;delete PThread.pthreads[pthread_ptr];PThread.unusedWorkers.push(worker);PThread.runningWorkers.splice(PThread.runningWorkers.indexOf(worker),1);worker.pthread_ptr=0;__emscripten_thread_free_data(pthread_ptr)},threadInitTLS(){PThread.tlsInitFunctions.forEach(f=>f())},loadWasmModuleToWorker:worker=>new Promise(onFinishedLoading=>{worker.onmessage=e=>{var d=e["data"];var cmd=d.cmd;if(d.targetThread&&d.targetThread!=_pthread_self()){var targetWorker=PThread.pthreads[d.targetThread];if(targetWorker){targetWorker.postMessage(d,d.transferList)}else{err(`Internal error! Worker sent a message "${cmd}" to target pthread ${d.targetThread}, but that thread no longer exists!`)}return}if(cmd==="checkMailbox"){checkMailbox()}else if(cmd==="spawnThread"){spawnThread(d)}else if(cmd==="cleanupThread"){callUserCallback(()=>cleanupThread(d.thread))}else if(cmd==="loaded"){worker.loaded=true;if(ENVIRONMENT_IS_NODE&&!worker.pthread_ptr){worker.unref()}onFinishedLoading(worker)}else if(d.target==="setimmediate"){worker.postMessage(d)}else if(cmd==="uncaughtException"){worker.onerror(d.error)}else if(cmd==="callHandler"){Module[d.handler](...d.args)}else if(cmd){err(`worker sent an unknown command ${cmd}`)}};worker.onerror=e=>{var message="worker sent an error!";err(`${message} ${e.filename}:${e.lineno}: ${e.message}`);throw e};if(ENVIRONMENT_IS_NODE){worker.on("message",data=>worker.onmessage({data}));worker.on("error",e=>worker.onerror(e))}var handlers=[];var knownHandlers=["onExit","onAbort","print","printErr"];for(var handler of knownHandlers){if(Module.propertyIsEnumerable(handler)){handlers.push(handler)}}worker.postMessage({cmd:"load",handlers,wasmMemory,wasmModule})}),async loadWasmModuleToAllWorkers(){if(ENVIRONMENT_IS_PTHREAD){return}let pthreadPoolReady=Promise.all(PThread.unusedWorkers.map(PThread.loadWasmModuleToWorker));return pthreadPoolReady},allocateUnusedWorker(){var worker;var pthreadMainJs=_scriptName;if(Module["mainScriptUrlOrBlob"]){pthreadMainJs=Module["mainScriptUrlOrBlob"];if(typeof pthreadMainJs!="string"){pthreadMainJs=URL.createObjectURL(pthreadMainJs)}}worker=new Worker(pthreadMainJs,{workerData:"em-pthread",name:"em-pthread"});PThread.unusedWorkers.push(worker)},getNewWorker(){if(PThread.unusedWorkers.length==0){PThread.allocateUnusedWorker();PThread.loadWasmModuleToWorker(PThread.unusedWorkers[0])}return PThread.unusedWorkers.pop()}};var onPostRuns=[];var addOnPostRun=cb=>onPostRuns.push(cb);function establishStackSpace(pthread_ptr){var stackHigh=Number((growMemViews(),HEAPU64)[(pthread_ptr+88)/8]);var stackSize=Number((growMemViews(),HEAPU64)[(pthread_ptr+96)/8]);var stackLow=stackHigh-stackSize;_emscripten_stack_set_limits(stackHigh,stackLow);stackRestore(stackHigh)}var wasmTableMirror=[];var getWasmTableEntry=funcPtr=>{funcPtr=Number(funcPtr);var func=wasmTableMirror[funcPtr];if(!func){wasmTableMirror[funcPtr]=func=wasmTable.get(BigInt(funcPtr));if(Asyncify.isAsyncExport(func)){wasmTableMirror[funcPtr]=func=Asyncify.makeAsyncFunction(func)}}return func};var invokeEntryPoint=async(ptr,arg)=>{runtimeKeepaliveCounter=0;noExitRuntime=0;var result=(a1=>WebAssembly.promising(getWasmTableEntry(ptr)).call(null,BigInt(a1)))(arg);function finish(result){if(keepRuntimeAlive()){EXITSTATUS=result;return}__emscripten_thread_exit(result)}result=await result;finish(result)};invokeEntryPoint.isAsync=true;var noExitRuntime=true;var registerTLSInit=tlsInitFunc=>PThread.tlsInitFunctions.push(tlsInitFunc);var wasmMemory;function pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(2,0,1,pthread_ptr,attr,startRoutine,arg);return ___pthread_create_js(pthread_ptr,attr,startRoutine,arg)}var _emscripten_has_threading_support=()=>!!globalThis.SharedArrayBuffer;var INT53_MAX=9007199254740992;var INT53_MIN=-9007199254740992;var bigintToI53Checked=num=>num<INT53_MIN||num>INT53_MAX?NaN:Number(num);function ___pthread_create_js(pthread_ptr,attr,startRoutine,arg){pthread_ptr=bigintToI53Checked(pthread_ptr);attr=bigintToI53Checked(attr);startRoutine=bigintToI53Checked(startRoutine);arg=bigintToI53Checked(arg);if(!_emscripten_has_threading_support()){return 6}var transferList=[];var error=0;if(ENVIRONMENT_IS_PTHREAD&&(transferList.length===0||error)){return pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg)}if(error)return error;var threadParams={startRoutine,pthread_ptr,arg,transferList};if(ENVIRONMENT_IS_PTHREAD){threadParams.cmd="spawnThread";postMessage(threadParams,transferList);return 0}return spawnThread(threadParams)}var syscallGetVarargP=()=>{var ret=Number((growMemViews(),HEAPU64)[SYSCALLS.varargs/8]);SYSCALLS.varargs+=8;return ret};var syscallGetVarargI=()=>{var ret=(growMemViews(),HEAP32)[+SYSCALLS.varargs/4];SYSCALLS.varargs+=4;return ret};var PATH={isAbs:path=>path.charAt(0)==="/",splitPath:filename=>{var splitPathRe=/^(\\/?|)([\\s\\S]*?)((?:\\.{1,2}|[^\\/]+?|)(\\.[^.\\/]*|))(?:[\\/]*)$/;return splitPathRe.exec(filename).slice(1)},normalizeArray:(parts,allowAboveRoot)=>{var up=0;for(var i=parts.length-1;i>=0;i--){var last=parts[i];if(last==="."){parts.splice(i,1)}else if(last===".."){parts.splice(i,1);up++}else if(up){parts.splice(i,1);up--}}if(allowAboveRoot){for(;up;up--){parts.unshift("..")}}return parts},normalize:path=>{var isAbsolute=PATH.isAbs(path),trailingSlash=path.slice(-1)==="/";path=PATH.normalizeArray(path.split("/").filter(p=>!!p),!isAbsolute).join("/");if(!path&&!isAbsolute){path="."}if(path&&trailingSlash){path+="/"}return(isAbsolute?"/":"")+path},dirname:path=>{var result=PATH.splitPath(path),root=result[0],dir=result[1];if(!root&&!dir){return"."}if(dir){dir=dir.slice(0,-1)}return root+dir},basename:path=>path&&path.match(/([^\\/]+|\\/)\\/*$/)[1],join:(...paths)=>PATH.normalize(paths.join("/")),join2:(l,r)=>PATH.normalize(l+"/"+r)};var initRandomFill=()=>view=>view.set(crypto.getRandomValues(new Uint8Array(view.byteLength)));var randomFill=view=>{(randomFill=initRandomFill())(view)};var PATH_FS={resolve:(...args)=>{var resolvedPath="",resolvedAbsolute=false;for(var i=args.length-1;i>=-1&&!resolvedAbsolute;i--){var path=i>=0?args[i]:FS.cwd();if(typeof path!="string"){throw new TypeError("Arguments to path.resolve must be strings")}else if(!path){return""}resolvedPath=path+"/"+resolvedPath;resolvedAbsolute=PATH.isAbs(path)}resolvedPath=PATH.normalizeArray(resolvedPath.split("/").filter(p=>!!p),!resolvedAbsolute).join("/");return(resolvedAbsolute?"/":"")+resolvedPath||"."},relative:(from,to)=>{from=PATH_FS.resolve(from).slice(1);to=PATH_FS.resolve(to).slice(1);function trim(arr){var start=0;for(;start<arr.length;start++){if(arr[start]!=="")break}var end=arr.length-1;for(;end>=0;end--){if(arr[end]!=="")break}if(start>end)return[];return arr.slice(start,end-start+1)}var fromParts=trim(from.split("/"));var toParts=trim(to.split("/"));var length=Math.min(fromParts.length,toParts.length);var samePartsLength=length;for(var i=0;i<length;i++){if(fromParts[i]!==toParts[i]){samePartsLength=i;break}}var outputParts=[];for(var i=samePartsLength;i<fromParts.length;i++){outputParts.push("..")}outputParts=outputParts.concat(toParts.slice(samePartsLength));return outputParts.join("/")}};var UTF8Decoder=globalThis.TextDecoder&&new TextDecoder;var findStringEnd=(heapOrArray,idx,maxBytesToRead,ignoreNul)=>{var maxIdx=idx+maxBytesToRead;if(ignoreNul)return maxIdx;while(heapOrArray[idx]&&!(idx>=maxIdx))++idx;return idx};var UTF8ArrayToString=(heapOrArray,idx=0,maxBytesToRead,ignoreNul)=>{var endPtr=findStringEnd(heapOrArray,idx,maxBytesToRead,ignoreNul);if(endPtr-idx>16&&heapOrArray.buffer&&UTF8Decoder){return UTF8Decoder.decode(heapOrArray.buffer instanceof ArrayBuffer?heapOrArray.subarray(idx,endPtr):heapOrArray.slice(idx,endPtr))}var str="";while(idx<endPtr){var u0=heapOrArray[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heapOrArray[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heapOrArray[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2}else{u0=(u0&7)<<18|u1<<12|u2<<6|heapOrArray[idx++]&63}if(u0<65536){str+=String.fromCharCode(u0)}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023)}}return str};var FS_stdin_getChar_buffer=[];var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++}else if(c<=2047){len+=2}else if(c>=55296&&c<=57343){len+=4;++i}else{len+=3}}return len};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.codePointAt(i);if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;i++}}heap[outIdx]=0;return outIdx-startIdx};var intArrayFromString=(stringy,dontAddNull,length)=>{var len=length>0?length:lengthBytesUTF8(stringy)+1;var u8array=new Array(len);var numBytesWritten=stringToUTF8Array(stringy,u8array,0,u8array.length);if(dontAddNull)u8array.length=numBytesWritten;return u8array};var FS_stdin_getChar=()=>{if(!FS_stdin_getChar_buffer.length){var result=null;if(ENVIRONMENT_IS_NODE){var BUFSIZE=256;var buf=Buffer.alloc(BUFSIZE);var bytesRead=0;var fd=process.stdin.fd;try{bytesRead=fs.readSync(fd,buf,0,BUFSIZE)}catch(e){if(e.toString().includes("EOF"))bytesRead=0;else throw e}if(bytesRead>0){result=buf.slice(0,bytesRead).toString("utf-8")}}else if(globalThis.window?.prompt){result=window.prompt("Input: ");if(result!==null){result+="\\n"}}else{}if(!result){return null}FS_stdin_getChar_buffer=intArrayFromString(result,true)}return FS_stdin_getChar_buffer.shift()};var TTY={ttys:[],init(){},shutdown(){},register(dev,ops){TTY.ttys[dev]={input:[],output:[],ops};FS.registerDevice(dev,TTY.stream_ops)},stream_ops:{open(stream){var tty=TTY.ttys[stream.node.rdev];if(!tty){throw new FS.ErrnoError(43)}stream.tty=tty;stream.seekable=false},close(stream){stream.tty.ops.fsync(stream.tty)},fsync(stream){stream.tty.ops.fsync(stream.tty)},read(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.get_char){throw new FS.ErrnoError(60)}var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=stream.tty.ops.get_char(stream.tty)}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.put_char){throw new FS.ErrnoError(60)}try{for(var i=0;i<length;i++){stream.tty.ops.put_char(stream.tty,buffer[offset+i])}}catch(e){throw new FS.ErrnoError(29)}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}},default_tty_ops:{get_char(tty){return FS_stdin_getChar()},put_char(tty,val){if(val===null||val===10){out(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){out(UTF8ArrayToString(tty.output));tty.output=[]}},ioctl_tcgets(tty){return{c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(tty,optional_actions,data){return 0},ioctl_tiocgwinsz(tty){return[24,80]}},default_tty1_ops:{put_char(tty,val){if(val===null||val===10){err(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){err(UTF8ArrayToString(tty.output));tty.output=[]}}}};var zeroMemory=(ptr,size)=>(growMemViews(),HEAPU8).fill(0,ptr,ptr+size);var alignMemory=(size,alignment)=>Math.ceil(size/alignment)*alignment;var mmapAlloc=size=>{size=alignMemory(size,65536);var ptr=_emscripten_builtin_memalign(65536,size);if(ptr)zeroMemory(ptr,size);return ptr};var MEMFS={ops_table:null,mount(mount){return MEMFS.createNode(null,"/",16895,0)},createNode(parent,name,mode,dev){if(FS.isBlkdev(mode)||FS.isFIFO(mode)){throw new FS.ErrnoError(63)}MEMFS.ops_table||={dir:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,lookup:MEMFS.node_ops.lookup,mknod:MEMFS.node_ops.mknod,rename:MEMFS.node_ops.rename,unlink:MEMFS.node_ops.unlink,rmdir:MEMFS.node_ops.rmdir,readdir:MEMFS.node_ops.readdir,symlink:MEMFS.node_ops.symlink},stream:{llseek:MEMFS.stream_ops.llseek}},file:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:{llseek:MEMFS.stream_ops.llseek,read:MEMFS.stream_ops.read,write:MEMFS.stream_ops.write,mmap:MEMFS.stream_ops.mmap,msync:MEMFS.stream_ops.msync}},link:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,readlink:MEMFS.node_ops.readlink},stream:{}},chrdev:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:FS.chrdev_stream_ops}};var node=FS.createNode(parent,name,mode,dev);if(FS.isDir(node.mode)){node.node_ops=MEMFS.ops_table.dir.node;node.stream_ops=MEMFS.ops_table.dir.stream;node.contents={}}else if(FS.isFile(node.mode)){node.node_ops=MEMFS.ops_table.file.node;node.stream_ops=MEMFS.ops_table.file.stream;node.usedBytes=0;node.contents=null}else if(FS.isLink(node.mode)){node.node_ops=MEMFS.ops_table.link.node;node.stream_ops=MEMFS.ops_table.link.stream}else if(FS.isChrdev(node.mode)){node.node_ops=MEMFS.ops_table.chrdev.node;node.stream_ops=MEMFS.ops_table.chrdev.stream}node.atime=node.mtime=node.ctime=Date.now();if(parent){parent.contents[name]=node;parent.atime=parent.mtime=parent.ctime=node.atime}return node},getFileDataAsTypedArray(node){if(!node.contents)return new Uint8Array(0);if(node.contents.subarray)return node.contents.subarray(0,node.usedBytes);return new Uint8Array(node.contents)},expandFileStorage(node,newCapacity){var prevCapacity=node.contents?node.contents.length:0;if(prevCapacity>=newCapacity)return;var CAPACITY_DOUBLING_MAX=1024*1024;newCapacity=Math.max(newCapacity,prevCapacity*(prevCapacity<CAPACITY_DOUBLING_MAX?2:1.125)>>>0);if(prevCapacity!=0)newCapacity=Math.max(newCapacity,256);var oldContents=node.contents;node.contents=new Uint8Array(newCapacity);if(node.usedBytes>0)node.contents.set(oldContents.subarray(0,node.usedBytes),0)},resizeFileStorage(node,newSize){if(node.usedBytes==newSize)return;if(newSize==0){node.contents=null;node.usedBytes=0}else{var oldContents=node.contents;node.contents=new Uint8Array(newSize);if(oldContents){node.contents.set(oldContents.subarray(0,Math.min(newSize,node.usedBytes)))}node.usedBytes=newSize}},node_ops:{getattr(node){var attr={};attr.dev=FS.isChrdev(node.mode)?node.id:1;attr.ino=node.id;attr.mode=node.mode;attr.nlink=1;attr.uid=0;attr.gid=0;attr.rdev=node.rdev;if(FS.isDir(node.mode)){attr.size=4096}else if(FS.isFile(node.mode)){attr.size=node.usedBytes}else if(FS.isLink(node.mode)){attr.size=node.link.length}else{attr.size=0}attr.atime=new Date(node.atime);attr.mtime=new Date(node.mtime);attr.ctime=new Date(node.ctime);attr.blksize=4096;attr.blocks=Math.ceil(attr.size/attr.blksize);return attr},setattr(node,attr){for(const key of["mode","atime","mtime","ctime"]){if(attr[key]!=null){node[key]=attr[key]}}if(attr.size!==undefined){MEMFS.resizeFileStorage(node,attr.size)}},lookup(parent,name){if(!MEMFS.doesNotExistError){MEMFS.doesNotExistError=new FS.ErrnoError(44);MEMFS.doesNotExistError.stack="<generic error, no stack>"}throw MEMFS.doesNotExistError},mknod(parent,name,mode,dev){return MEMFS.createNode(parent,name,mode,dev)},rename(old_node,new_dir,new_name){var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(new_node){if(FS.isDir(old_node.mode)){for(var i in new_node.contents){throw new FS.ErrnoError(55)}}FS.hashRemoveNode(new_node)}delete old_node.parent.contents[old_node.name];new_dir.contents[new_name]=old_node;old_node.name=new_name;new_dir.ctime=new_dir.mtime=old_node.parent.ctime=old_node.parent.mtime=Date.now()},unlink(parent,name){delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},rmdir(parent,name){var node=FS.lookupNode(parent,name);for(var i in node.contents){throw new FS.ErrnoError(55)}delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},readdir(node){return[".","..",...Object.keys(node.contents)]},symlink(parent,newname,oldpath){var node=MEMFS.createNode(parent,newname,511|40960,0);node.link=oldpath;return node},readlink(node){if(!FS.isLink(node.mode)){throw new FS.ErrnoError(28)}return node.link}},stream_ops:{read(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=stream.node.usedBytes)return 0;var size=Math.min(stream.node.usedBytes-position,length);if(size>8&&contents.subarray){buffer.set(contents.subarray(position,position+size),offset)}else{for(var i=0;i<size;i++)buffer[offset+i]=contents[position+i]}return size},write(stream,buffer,offset,length,position,canOwn){if(buffer.buffer===(growMemViews(),HEAP8).buffer){canOwn=false}if(!length)return 0;var node=stream.node;node.mtime=node.ctime=Date.now();if(buffer.subarray&&(!node.contents||node.contents.subarray)){if(canOwn){node.contents=buffer.subarray(offset,offset+length);node.usedBytes=length;return length}else if(node.usedBytes===0&&position===0){node.contents=buffer.slice(offset,offset+length);node.usedBytes=length;return length}else if(position+length<=node.usedBytes){node.contents.set(buffer.subarray(offset,offset+length),position);return length}}MEMFS.expandFileStorage(node,position+length);if(node.contents.subarray&&buffer.subarray){node.contents.set(buffer.subarray(offset,offset+length),position)}else{for(var i=0;i<length;i++){node.contents[position+i]=buffer[offset+i]}}node.usedBytes=Math.max(node.usedBytes,position+length);return length},llseek(stream,offset,whence){var position=offset;if(whence===1){position+=stream.position}else if(whence===2){if(FS.isFile(stream.node.mode)){position+=stream.node.usedBytes}}if(position<0){throw new FS.ErrnoError(28)}return position},mmap(stream,length,position,prot,flags){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}var ptr;var allocated;var contents=stream.node.contents;if(!(flags&2)&&contents&&contents.buffer===(growMemViews(),HEAP8).buffer){allocated=false;ptr=contents.byteOffset}else{allocated=true;ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}if(contents){if(position>0||position+length<contents.length){if(contents.subarray){contents=contents.subarray(position,position+length)}else{contents=Array.prototype.slice.call(contents,position,position+length)}}(growMemViews(),HEAP8).set(contents,ptr)}}return{ptr,allocated}},msync(stream,buffer,offset,length,mmapFlags){MEMFS.stream_ops.write(stream,buffer,0,length,offset,false);return 0}}};var FS_modeStringToFlags=str=>{var flagModes={r:0,"r+":2,w:512|64|1,"w+":512|64|2,a:1024|64|1,"a+":1024|64|2};var flags=flagModes[str];if(typeof flags=="undefined"){throw new Error(`Unknown file open mode: ${str}`)}return flags};var FS_getMode=(canRead,canWrite)=>{var mode=0;if(canRead)mode|=292|73;if(canWrite)mode|=146;return mode};var asyncLoad=async url=>{var arrayBuffer=await readAsync(url);return new Uint8Array(arrayBuffer)};var FS_createDataFile=(...args)=>FS.createDataFile(...args);var getUniqueRunDependency=id=>id;var preloadPlugins=[];var FS_handledByPreloadPlugin=async(byteArray,fullname)=>{if(typeof Browser!="undefined")Browser.init();for(var plugin of preloadPlugins){if(plugin["canHandle"](fullname)){return plugin["handle"](byteArray,fullname)}}return byteArray};var FS_preloadFile=async(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish)=>{var fullname=name?PATH_FS.resolve(PATH.join2(parent,name)):parent;var dep=getUniqueRunDependency(`cp ${fullname}`);addRunDependency(dep);try{var byteArray=url;if(typeof url=="string"){byteArray=await asyncLoad(url)}byteArray=await FS_handledByPreloadPlugin(byteArray,fullname);preFinish?.();if(!dontCreateFile){FS_createDataFile(parent,name,byteArray,canRead,canWrite,canOwn)}}finally{removeRunDependency(dep)}};var FS_createPreloadedFile=(parent,name,url,canRead,canWrite,onload,onerror,dontCreateFile,canOwn,preFinish)=>{FS_preloadFile(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish).then(onload).catch(onerror)};var FS={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,filesystems:null,syncFSRequests:0,readFiles:{},ErrnoError:class{name="ErrnoError";constructor(errno){this.errno=errno}},FSStream:class{shared={};get object(){return this.node}set object(val){this.node=val}get isRead(){return(this.flags&2097155)!==1}get isWrite(){return(this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(val){this.shared.flags=val}get position(){return this.shared.position}set position(val){this.shared.position=val}},FSNode:class{node_ops={};stream_ops={};readMode=292|73;writeMode=146;mounted=null;constructor(parent,name,mode,rdev){if(!parent){parent=this}this.parent=parent;this.mount=parent.mount;this.id=FS.nextInode++;this.name=name;this.mode=mode;this.rdev=rdev;this.atime=this.mtime=this.ctime=Date.now()}get read(){return(this.mode&this.readMode)===this.readMode}set read(val){val?this.mode|=this.readMode:this.mode&=~this.readMode}get write(){return(this.mode&this.writeMode)===this.writeMode}set write(val){val?this.mode|=this.writeMode:this.mode&=~this.writeMode}get isFolder(){return FS.isDir(this.mode)}get isDevice(){return FS.isChrdev(this.mode)}},lookupPath(path,opts={}){if(!path){throw new FS.ErrnoError(44)}opts.follow_mount??=true;if(!PATH.isAbs(path)){path=FS.cwd()+"/"+path}linkloop:for(var nlinks=0;nlinks<40;nlinks++){var parts=path.split("/").filter(p=>!!p);var current=FS.root;var current_path="/";for(var i=0;i<parts.length;i++){var islast=i===parts.length-1;if(islast&&opts.parent){break}if(parts[i]==="."){continue}if(parts[i]===".."){current_path=PATH.dirname(current_path);if(FS.isRoot(current)){path=current_path+"/"+parts.slice(i+1).join("/");nlinks--;continue linkloop}else{current=current.parent}continue}current_path=PATH.join2(current_path,parts[i]);try{current=FS.lookupNode(current,parts[i])}catch(e){if(e?.errno===44&&islast&&opts.noent_okay){return{path:current_path}}throw e}if(FS.isMountpoint(current)&&(!islast||opts.follow_mount)){current=current.mounted.root}if(FS.isLink(current.mode)&&(!islast||opts.follow)){if(!current.node_ops.readlink){throw new FS.ErrnoError(52)}var link=current.node_ops.readlink(current);if(!PATH.isAbs(link)){link=PATH.dirname(current_path)+"/"+link}path=link+"/"+parts.slice(i+1).join("/");continue linkloop}}return{path:current_path,node:current}}throw new FS.ErrnoError(32)},getPath(node){var path;while(true){if(FS.isRoot(node)){var mount=node.mount.mountpoint;if(!path)return mount;return mount[mount.length-1]!=="/"?`${mount}/${path}`:mount+path}path=path?`${node.name}/${path}`:node.name;node=node.parent}},hashName(parentid,name){var hash=0;for(var i=0;i<name.length;i++){hash=(hash<<5)-hash+name.charCodeAt(i)|0}return(parentid+hash>>>0)%FS.nameTable.length},hashAddNode(node){var hash=FS.hashName(node.parent.id,node.name);node.name_next=FS.nameTable[hash];FS.nameTable[hash]=node},hashRemoveNode(node){var hash=FS.hashName(node.parent.id,node.name);if(FS.nameTable[hash]===node){FS.nameTable[hash]=node.name_next}else{var current=FS.nameTable[hash];while(current){if(current.name_next===node){current.name_next=node.name_next;break}current=current.name_next}}},lookupNode(parent,name){var errCode=FS.mayLookup(parent);if(errCode){throw new FS.ErrnoError(errCode)}var hash=FS.hashName(parent.id,name);for(var node=FS.nameTable[hash];node;node=node.name_next){var nodeName=node.name;if(node.parent.id===parent.id&&nodeName===name){return node}}return FS.lookup(parent,name)},createNode(parent,name,mode,rdev){var node=new FS.FSNode(parent,name,mode,rdev);FS.hashAddNode(node);return node},destroyNode(node){FS.hashRemoveNode(node)},isRoot(node){return node===node.parent},isMountpoint(node){return!!node.mounted},isFile(mode){return(mode&61440)===32768},isDir(mode){return(mode&61440)===16384},isLink(mode){return(mode&61440)===40960},isChrdev(mode){return(mode&61440)===8192},isBlkdev(mode){return(mode&61440)===24576},isFIFO(mode){return(mode&61440)===4096},isSocket(mode){return(mode&49152)===49152},flagsToPermissionString(flag){var perms=["r","w","rw"][flag&3];if(flag&512){perms+="w"}return perms},nodePermissions(node,perms){if(FS.ignorePermissions){return 0}if(perms.includes("r")&&!(node.mode&292)){return 2}else if(perms.includes("w")&&!(node.mode&146)){return 2}else if(perms.includes("x")&&!(node.mode&73)){return 2}return 0},mayLookup(dir){if(!FS.isDir(dir.mode))return 54;var errCode=FS.nodePermissions(dir,"x");if(errCode)return errCode;if(!dir.node_ops.lookup)return 2;return 0},mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")},mayDelete(dir,name,isdir){var node;try{node=FS.lookupNode(dir,name)}catch(e){return e.errno}var errCode=FS.nodePermissions(dir,"wx");if(errCode){return errCode}if(isdir){if(!FS.isDir(node.mode)){return 54}if(FS.isRoot(node)||FS.getPath(node)===FS.cwd()){return 10}}else{if(FS.isDir(node.mode)){return 31}}return 0},mayOpen(node,flags){if(!node){return 44}if(FS.isLink(node.mode)){return 32}else if(FS.isDir(node.mode)){if(FS.flagsToPermissionString(flags)!=="r"||flags&(512|64)){return 31}}return FS.nodePermissions(node,FS.flagsToPermissionString(flags))},checkOpExists(op,err){if(!op){throw new FS.ErrnoError(err)}return op},MAX_OPEN_FDS:4096,nextfd(){for(var fd=0;fd<=FS.MAX_OPEN_FDS;fd++){if(!FS.streams[fd]){return fd}}throw new FS.ErrnoError(33)},getStreamChecked(fd){var stream=FS.getStream(fd);if(!stream){throw new FS.ErrnoError(8)}return stream},getStream:fd=>FS.streams[fd],createStream(stream,fd=-1){stream=Object.assign(new FS.FSStream,stream);if(fd==-1){fd=FS.nextfd()}stream.fd=fd;FS.streams[fd]=stream;return stream},closeStream(fd){FS.streams[fd]=null},dupStream(origStream,fd=-1){var stream=FS.createStream(origStream,fd);stream.stream_ops?.dup?.(stream);return stream},doSetAttr(stream,node,attr){var setattr=stream?.stream_ops.setattr;var arg=setattr?stream:node;setattr??=node.node_ops.setattr;FS.checkOpExists(setattr,63);setattr(arg,attr)},chrdev_stream_ops:{open(stream){var device=FS.getDevice(stream.node.rdev);stream.stream_ops=device.stream_ops;stream.stream_ops.open?.(stream)},llseek(){throw new FS.ErrnoError(70)}},major:dev=>dev>>8,minor:dev=>dev&255,makedev:(ma,mi)=>ma<<8|mi,registerDevice(dev,ops){FS.devices[dev]={stream_ops:ops}},getDevice:dev=>FS.devices[dev],getMounts(mount){var mounts=[];var check=[mount];while(check.length){var m=check.pop();mounts.push(m);check.push(...m.mounts)}return mounts},syncfs(populate,callback){if(typeof populate=="function"){callback=populate;populate=false}FS.syncFSRequests++;if(FS.syncFSRequests>1){err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`)}var mounts=FS.getMounts(FS.root.mount);var completed=0;function doCallback(errCode){FS.syncFSRequests--;return callback(errCode)}function done(errCode){if(errCode){if(!done.errored){done.errored=true;return doCallback(errCode)}return}if(++completed>=mounts.length){doCallback(null)}}for(var mount of mounts){if(mount.type.syncfs){mount.type.syncfs(mount,populate,done)}else{done(null)}}},mount(type,opts,mountpoint){var root=mountpoint==="/";var pseudo=!mountpoint;var node;if(root&&FS.root){throw new FS.ErrnoError(10)}else if(!root&&!pseudo){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});mountpoint=lookup.path;node=lookup.node;if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}if(!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}}var mount={type,opts,mountpoint,mounts:[]};var mountRoot=type.mount(mount);mountRoot.mount=mount;mount.root=mountRoot;if(root){FS.root=mountRoot}else if(node){node.mounted=mount;if(node.mount){node.mount.mounts.push(mount)}}return mountRoot},unmount(mountpoint){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});if(!FS.isMountpoint(lookup.node)){throw new FS.ErrnoError(28)}var node=lookup.node;var mount=node.mounted;var mounts=FS.getMounts(mount);for(var[hash,current]of Object.entries(FS.nameTable)){while(current){var next=current.name_next;if(mounts.includes(current.mount)){FS.destroyNode(current)}current=next}}node.mounted=null;var idx=node.mount.mounts.indexOf(mount);node.mount.mounts.splice(idx,1)},lookup(parent,name){return parent.node_ops.lookup(parent,name)},mknod(path,mode,dev){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);if(!name){throw new FS.ErrnoError(28)}if(name==="."||name===".."){throw new FS.ErrnoError(20)}var errCode=FS.mayCreate(parent,name);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.mknod){throw new FS.ErrnoError(63)}return parent.node_ops.mknod(parent,name,mode,dev)},statfs(path){return FS.statfsNode(FS.lookupPath(path,{follow:true}).node)},statfsStream(stream){return FS.statfsNode(stream.node)},statfsNode(node){var rtn={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:FS.nextInode,ffree:FS.nextInode-1,fsid:42,flags:2,namelen:255};if(node.node_ops.statfs){Object.assign(rtn,node.node_ops.statfs(node.mount.opts.root))}return rtn},create(path,mode=438){mode&=4095;mode|=32768;return FS.mknod(path,mode,0)},mkdir(path,mode=511){mode&=511|512;mode|=16384;return FS.mknod(path,mode,0)},mkdirTree(path,mode){var dirs=path.split("/");var d="";for(var dir of dirs){if(!dir)continue;if(d||PATH.isAbs(path))d+="/";d+=dir;try{FS.mkdir(d,mode)}catch(e){if(e.errno!=20)throw e}}},mkdev(path,mode,dev){if(typeof dev=="undefined"){dev=mode;mode=438}mode|=8192;return FS.mknod(path,mode,dev)},symlink(oldpath,newpath){if(!PATH_FS.resolve(oldpath)){throw new FS.ErrnoError(44)}var lookup=FS.lookupPath(newpath,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var newname=PATH.basename(newpath);var errCode=FS.mayCreate(parent,newname);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.symlink){throw new FS.ErrnoError(63)}return parent.node_ops.symlink(parent,newname,oldpath)},rename(old_path,new_path){var old_dirname=PATH.dirname(old_path);var new_dirname=PATH.dirname(new_path);var old_name=PATH.basename(old_path);var new_name=PATH.basename(new_path);var lookup,old_dir,new_dir;lookup=FS.lookupPath(old_path,{parent:true});old_dir=lookup.node;lookup=FS.lookupPath(new_path,{parent:true});new_dir=lookup.node;if(!old_dir||!new_dir)throw new FS.ErrnoError(44);if(old_dir.mount!==new_dir.mount){throw new FS.ErrnoError(75)}var old_node=FS.lookupNode(old_dir,old_name);var relative=PATH_FS.relative(old_path,new_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(28)}relative=PATH_FS.relative(new_path,old_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(55)}var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(old_node===new_node){return}var isdir=FS.isDir(old_node.mode);var errCode=FS.mayDelete(old_dir,old_name,isdir);if(errCode){throw new FS.ErrnoError(errCode)}errCode=new_node?FS.mayDelete(new_dir,new_name,isdir):FS.mayCreate(new_dir,new_name);if(errCode){throw new FS.ErrnoError(errCode)}if(!old_dir.node_ops.rename){throw new FS.ErrnoError(63)}if(FS.isMountpoint(old_node)||new_node&&FS.isMountpoint(new_node)){throw new FS.ErrnoError(10)}if(new_dir!==old_dir){errCode=FS.nodePermissions(old_dir,"w");if(errCode){throw new FS.ErrnoError(errCode)}}FS.hashRemoveNode(old_node);try{old_dir.node_ops.rename(old_node,new_dir,new_name);old_node.parent=new_dir}catch(e){throw e}finally{FS.hashAddNode(old_node)}},rmdir(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,true);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.rmdir){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.rmdir(parent,name);FS.destroyNode(node)},readdir(path){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var readdir=FS.checkOpExists(node.node_ops.readdir,54);return readdir(node)},unlink(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,false);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.unlink){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.unlink(parent,name);FS.destroyNode(node)},readlink(path){var lookup=FS.lookupPath(path);var link=lookup.node;if(!link){throw new FS.ErrnoError(44)}if(!link.node_ops.readlink){throw new FS.ErrnoError(28)}return link.node_ops.readlink(link)},stat(path,dontFollow){var lookup=FS.lookupPath(path,{follow:!dontFollow});var node=lookup.node;var getattr=FS.checkOpExists(node.node_ops.getattr,63);return getattr(node)},fstat(fd){var stream=FS.getStreamChecked(fd);var node=stream.node;var getattr=stream.stream_ops.getattr;var arg=getattr?stream:node;getattr??=node.node_ops.getattr;FS.checkOpExists(getattr,63);return getattr(arg)},lstat(path){return FS.stat(path,true)},doChmod(stream,node,mode,dontFollow){FS.doSetAttr(stream,node,{mode:mode&4095|node.mode&~4095,ctime:Date.now(),dontFollow})},chmod(path,mode,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChmod(null,node,mode,dontFollow)},lchmod(path,mode){FS.chmod(path,mode,true)},fchmod(fd,mode){var stream=FS.getStreamChecked(fd);FS.doChmod(stream,stream.node,mode,false)},doChown(stream,node,dontFollow){FS.doSetAttr(stream,node,{timestamp:Date.now(),dontFollow})},chown(path,uid,gid,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChown(null,node,dontFollow)},lchown(path,uid,gid){FS.chown(path,uid,gid,true)},fchown(fd,uid,gid){var stream=FS.getStreamChecked(fd);FS.doChown(stream,stream.node,false)},doTruncate(stream,node,len){if(FS.isDir(node.mode)){throw new FS.ErrnoError(31)}if(!FS.isFile(node.mode)){throw new FS.ErrnoError(28)}var errCode=FS.nodePermissions(node,"w");if(errCode){throw new FS.ErrnoError(errCode)}FS.doSetAttr(stream,node,{size:len,timestamp:Date.now()})},truncate(path,len){if(len<0){throw new FS.ErrnoError(28)}var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:true});node=lookup.node}else{node=path}FS.doTruncate(null,node,len)},ftruncate(fd,len){var stream=FS.getStreamChecked(fd);if(len<0||(stream.flags&2097155)===0){throw new FS.ErrnoError(28)}FS.doTruncate(stream,stream.node,len)},utime(path,atime,mtime){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var setattr=FS.checkOpExists(node.node_ops.setattr,63);setattr(node,{atime,mtime})},open(path,flags,mode=438){if(path===""){throw new FS.ErrnoError(44)}flags=typeof flags=="string"?FS_modeStringToFlags(flags):flags;if(flags&64){mode=mode&4095|32768}else{mode=0}var node;var isDirPath;if(typeof path=="object"){node=path}else{isDirPath=path.endsWith("/");var lookup=FS.lookupPath(path,{follow:!(flags&131072),noent_okay:true});node=lookup.node;path=lookup.path}var created=false;if(flags&64){if(node){if(flags&128){throw new FS.ErrnoError(20)}}else if(isDirPath){throw new FS.ErrnoError(31)}else{node=FS.mknod(path,mode|511,0);created=true}}if(!node){throw new FS.ErrnoError(44)}if(FS.isChrdev(node.mode)){flags&=~512}if(flags&65536&&!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}if(!created){var errCode=FS.mayOpen(node,flags);if(errCode){throw new FS.ErrnoError(errCode)}}if(flags&512&&!created){FS.truncate(node,0)}flags&=~(128|512|131072);var stream=FS.createStream({node,path:FS.getPath(node),flags,seekable:true,position:0,stream_ops:node.stream_ops,ungotten:[],error:false});if(stream.stream_ops.open){stream.stream_ops.open(stream)}if(created){FS.chmod(node,mode&511)}if(Module["logReadFiles"]&&!(flags&1)){if(!(path in FS.readFiles)){FS.readFiles[path]=1}}return stream},close(stream){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(stream.getdents)stream.getdents=null;try{if(stream.stream_ops.close){stream.stream_ops.close(stream)}}catch(e){throw e}finally{FS.closeStream(stream.fd)}stream.fd=null},isClosed(stream){return stream.fd===null},llseek(stream,offset,whence){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(!stream.seekable||!stream.stream_ops.llseek){throw new FS.ErrnoError(70)}if(whence!=0&&whence!=1&&whence!=2){throw new FS.ErrnoError(28)}stream.position=stream.stream_ops.llseek(stream,offset,whence);stream.ungotten=[];return stream.position},read(stream,buffer,offset,length,position){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.read){throw new FS.ErrnoError(28)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesRead=stream.stream_ops.read(stream,buffer,offset,length,position);if(!seeking)stream.position+=bytesRead;return bytesRead},write(stream,buffer,offset,length,position,canOwn){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===0){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.write){throw new FS.ErrnoError(28)}if(stream.seekable&&stream.flags&1024){FS.llseek(stream,0,2)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesWritten=stream.stream_ops.write(stream,buffer,offset,length,position,canOwn);if(!seeking)stream.position+=bytesWritten;return bytesWritten},mmap(stream,length,position,prot,flags){if((prot&2)!==0&&(flags&2)===0&&(stream.flags&2097155)!==2){throw new FS.ErrnoError(2)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(2)}if(!stream.stream_ops.mmap){throw new FS.ErrnoError(43)}if(!length){throw new FS.ErrnoError(28)}return stream.stream_ops.mmap(stream,length,position,prot,flags)},msync(stream,buffer,offset,length,mmapFlags){if(!stream.stream_ops.msync){return 0}return stream.stream_ops.msync(stream,buffer,offset,length,mmapFlags)},ioctl(stream,cmd,arg){if(!stream.stream_ops.ioctl){throw new FS.ErrnoError(59)}return stream.stream_ops.ioctl(stream,cmd,arg)},readFile(path,opts={}){opts.flags=opts.flags||0;opts.encoding=opts.encoding||"binary";if(opts.encoding!=="utf8"&&opts.encoding!=="binary"){abort(`Invalid encoding type "${opts.encoding}"`)}var stream=FS.open(path,opts.flags);var stat=FS.stat(path);var length=stat.size;var buf=new Uint8Array(length);FS.read(stream,buf,0,length,0);if(opts.encoding==="utf8"){buf=UTF8ArrayToString(buf)}FS.close(stream);return buf},writeFile(path,data,opts={}){opts.flags=opts.flags||577;var stream=FS.open(path,opts.flags,opts.mode);if(typeof data=="string"){data=new Uint8Array(intArrayFromString(data,true))}if(ArrayBuffer.isView(data)){FS.write(stream,data,0,data.byteLength,undefined,opts.canOwn)}else{abort("Unsupported data type")}FS.close(stream)},cwd:()=>FS.currentPath,chdir(path){var lookup=FS.lookupPath(path,{follow:true});if(lookup.node===null){throw new FS.ErrnoError(44)}if(!FS.isDir(lookup.node.mode)){throw new FS.ErrnoError(54)}var errCode=FS.nodePermissions(lookup.node,"x");if(errCode){throw new FS.ErrnoError(errCode)}FS.currentPath=lookup.path},createDefaultDirectories(){FS.mkdir("/tmp");FS.mkdir("/home");FS.mkdir("/home/web_user")},createDefaultDevices(){FS.mkdir("/dev");FS.registerDevice(FS.makedev(1,3),{read:()=>0,write:(stream,buffer,offset,length,pos)=>length,llseek:()=>0});FS.mkdev("/dev/null",FS.makedev(1,3));TTY.register(FS.makedev(5,0),TTY.default_tty_ops);TTY.register(FS.makedev(6,0),TTY.default_tty1_ops);FS.mkdev("/dev/tty",FS.makedev(5,0));FS.mkdev("/dev/tty1",FS.makedev(6,0));var randomBuffer=new Uint8Array(1024),randomLeft=0;var randomByte=()=>{if(randomLeft===0){randomFill(randomBuffer);randomLeft=randomBuffer.byteLength}return randomBuffer[--randomLeft]};FS.createDevice("/dev","random",randomByte);FS.createDevice("/dev","urandom",randomByte);FS.mkdir("/dev/shm");FS.mkdir("/dev/shm/tmp")},createSpecialDirectories(){FS.mkdir("/proc");var proc_self=FS.mkdir("/proc/self");FS.mkdir("/proc/self/fd");FS.mount({mount(){var node=FS.createNode(proc_self,"fd",16895,73);node.stream_ops={llseek:MEMFS.stream_ops.llseek};node.node_ops={lookup(parent,name){var fd=+name;var stream=FS.getStreamChecked(fd);var ret={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>stream.path},id:fd+1};ret.parent=ret;return ret},readdir(){return Array.from(FS.streams.entries()).filter(([k,v])=>v).map(([k,v])=>k.toString())}};return node}},{},"/proc/self/fd")},createStandardStreams(input,output,error){if(input){FS.createDevice("/dev","stdin",input)}else{FS.symlink("/dev/tty","/dev/stdin")}if(output){FS.createDevice("/dev","stdout",null,output)}else{FS.symlink("/dev/tty","/dev/stdout")}if(error){FS.createDevice("/dev","stderr",null,error)}else{FS.symlink("/dev/tty1","/dev/stderr")}var stdin=FS.open("/dev/stdin",0);var stdout=FS.open("/dev/stdout",1);var stderr=FS.open("/dev/stderr",1)},staticInit(){FS.nameTable=new Array(4096);FS.mount(MEMFS,{},"/");FS.createDefaultDirectories();FS.createDefaultDevices();FS.createSpecialDirectories();FS.filesystems={MEMFS}},init(input,output,error){FS.initialized=true;input??=Module["stdin"];output??=Module["stdout"];error??=Module["stderr"];FS.createStandardStreams(input,output,error)},quit(){FS.initialized=false;for(var stream of FS.streams){if(stream){FS.close(stream)}}},findObject(path,dontResolveLastLink){var ret=FS.analyzePath(path,dontResolveLastLink);if(!ret.exists){return null}return ret.object},analyzePath(path,dontResolveLastLink){try{var lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});path=lookup.path}catch(e){}var ret={isRoot:false,exists:false,error:0,name:null,path:null,object:null,parentExists:false,parentPath:null,parentObject:null};try{var lookup=FS.lookupPath(path,{parent:true});ret.parentExists=true;ret.parentPath=lookup.path;ret.parentObject=lookup.node;ret.name=PATH.basename(path);lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});ret.exists=true;ret.path=lookup.path;ret.object=lookup.node;ret.name=lookup.node.name;ret.isRoot=lookup.path==="/"}catch(e){ret.error=e.errno}return ret},createPath(parent,path,canRead,canWrite){parent=typeof parent=="string"?parent:FS.getPath(parent);var parts=path.split("/").reverse();while(parts.length){var part=parts.pop();if(!part)continue;var current=PATH.join2(parent,part);try{FS.mkdir(current)}catch(e){if(e.errno!=20)throw e}parent=current}return current},createFile(parent,name,properties,canRead,canWrite){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(canRead,canWrite);return FS.create(path,mode)},createDataFile(parent,name,data,canRead,canWrite,canOwn){var path=name;if(parent){parent=typeof parent=="string"?parent:FS.getPath(parent);path=name?PATH.join2(parent,name):parent}var mode=FS_getMode(canRead,canWrite);var node=FS.create(path,mode);if(data){if(typeof data=="string"){var arr=new Array(data.length);for(var i=0,len=data.length;i<len;++i)arr[i]=data.charCodeAt(i);data=arr}FS.chmod(node,mode|146);var stream=FS.open(node,577);FS.write(stream,data,0,data.length,0,canOwn);FS.close(stream);FS.chmod(node,mode)}},createDevice(parent,name,input,output){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(!!input,!!output);FS.createDevice.major??=64;var dev=FS.makedev(FS.createDevice.major++,0);FS.registerDevice(dev,{open(stream){stream.seekable=false},close(stream){if(output?.buffer?.length){output(10)}},read(stream,buffer,offset,length,pos){var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=input()}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){for(var i=0;i<length;i++){try{output(buffer[offset+i])}catch(e){throw new FS.ErrnoError(29)}}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}});return FS.mkdev(path,mode,dev)},forceLoadFile(obj){if(obj.isDevice||obj.isFolder||obj.link||obj.contents)return true;if(globalThis.XMLHttpRequest){abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.")}else{try{obj.contents=readBinary(obj.url)}catch(e){throw new FS.ErrnoError(29)}}},createLazyFile(parent,name,url,canRead,canWrite){class LazyUint8Array{lengthKnown=false;chunks=[];get(idx){if(idx>this.length-1||idx<0){return undefined}var chunkOffset=idx%this.chunkSize;var chunkNum=idx/this.chunkSize|0;return this.getter(chunkNum)[chunkOffset]}setDataGetter(getter){this.getter=getter}cacheLength(){var xhr=new XMLHttpRequest;xhr.open("HEAD",url,false);xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);var datalength=Number(xhr.getResponseHeader("Content-length"));var header;var hasByteServing=(header=xhr.getResponseHeader("Accept-Ranges"))&&header==="bytes";var usesGzip=(header=xhr.getResponseHeader("Content-Encoding"))&&header==="gzip";var chunkSize=1024*1024;if(!hasByteServing)chunkSize=datalength;var doXHR=(from,to)=>{if(from>to)abort("invalid range ("+from+", "+to+") or no bytes requested!");if(to>datalength-1)abort("only "+datalength+" bytes available! programmer error!");var xhr=new XMLHttpRequest;xhr.open("GET",url,false);if(datalength!==chunkSize)xhr.setRequestHeader("Range","bytes="+from+"-"+to);xhr.responseType="arraybuffer";if(xhr.overrideMimeType){xhr.overrideMimeType("text/plain; charset=x-user-defined")}xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);if(xhr.response!==undefined){return new Uint8Array(xhr.response||[])}return intArrayFromString(xhr.responseText||"",true)};var lazyArray=this;lazyArray.setDataGetter(chunkNum=>{var start=chunkNum*chunkSize;var end=(chunkNum+1)*chunkSize-1;end=Math.min(end,datalength-1);if(typeof lazyArray.chunks[chunkNum]=="undefined"){lazyArray.chunks[chunkNum]=doXHR(start,end)}if(typeof lazyArray.chunks[chunkNum]=="undefined")abort("doXHR failed!");return lazyArray.chunks[chunkNum]});if(usesGzip||!datalength){chunkSize=datalength=1;datalength=this.getter(0).length;chunkSize=datalength;out("LazyFiles on gzip forces download of the whole file when length is accessed")}this._length=datalength;this._chunkSize=chunkSize;this.lengthKnown=true}get length(){if(!this.lengthKnown){this.cacheLength()}return this._length}get chunkSize(){if(!this.lengthKnown){this.cacheLength()}return this._chunkSize}}if(globalThis.XMLHttpRequest){if(!ENVIRONMENT_IS_WORKER)abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");var lazyArray=new LazyUint8Array;var properties={isDevice:false,contents:lazyArray}}else{var properties={isDevice:false,url}}var node=FS.createFile(parent,name,properties,canRead,canWrite);if(properties.contents){node.contents=properties.contents}else if(properties.url){node.contents=null;node.url=properties.url}Object.defineProperties(node,{usedBytes:{get:function(){return this.contents.length}}});var stream_ops={};for(const[key,fn]of Object.entries(node.stream_ops)){stream_ops[key]=(...args)=>{FS.forceLoadFile(node);return fn(...args)}}function writeChunks(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=contents.length)return 0;var size=Math.min(contents.length-position,length);if(contents.slice){for(var i=0;i<size;i++){buffer[offset+i]=contents[position+i]}}else{for(var i=0;i<size;i++){buffer[offset+i]=contents.get(position+i)}}return size}stream_ops.read=(stream,buffer,offset,length,position)=>{FS.forceLoadFile(node);return writeChunks(stream,buffer,offset,length,position)};stream_ops.mmap=(stream,length,position,prot,flags)=>{FS.forceLoadFile(node);var ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}writeChunks(stream,(growMemViews(),HEAP8),ptr,length,position);return{ptr,allocated:true}};node.stream_ops=stream_ops;return node}};var UTF8ToString=(ptr,maxBytesToRead,ignoreNul)=>ptr?UTF8ArrayToString((growMemViews(),HEAPU8),ptr,maxBytesToRead,ignoreNul):"";var SYSCALLS={DEFAULT_POLLMASK:5,calculateAt(dirfd,path,allowEmpty){if(PATH.isAbs(path)){return path}var dir;if(dirfd===-100){dir=FS.cwd()}else{var dirstream=SYSCALLS.getStreamFromFD(dirfd);dir=dirstream.path}if(path.length==0){if(!allowEmpty){throw new FS.ErrnoError(44)}return dir}return dir+"/"+path},writeStat(buf,stat){(growMemViews(),HEAPU32)[buf/4]=stat.dev;(growMemViews(),HEAPU32)[(buf+4)/4]=stat.mode;(growMemViews(),HEAPU64)[(buf+8)/8]=BigInt(stat.nlink);(growMemViews(),HEAPU32)[(buf+16)/4]=stat.uid;(growMemViews(),HEAPU32)[(buf+20)/4]=stat.gid;(growMemViews(),HEAPU32)[(buf+24)/4]=stat.rdev;(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stat.size);(growMemViews(),HEAP32)[(buf+40)/4]=4096;(growMemViews(),HEAP32)[(buf+44)/4]=stat.blocks;var atime=stat.atime.getTime();var mtime=stat.mtime.getTime();var ctime=stat.ctime.getTime();(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(Math.floor(atime/1e3));(growMemViews(),HEAPU64)[(buf+56)/8]=BigInt(atime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+64)/8]=BigInt(Math.floor(mtime/1e3));(growMemViews(),HEAPU64)[(buf+72)/8]=BigInt(mtime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+80)/8]=BigInt(Math.floor(ctime/1e3));(growMemViews(),HEAPU64)[(buf+88)/8]=BigInt(ctime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+96)/8]=BigInt(stat.ino);return 0},writeStatFs(buf,stats){(growMemViews(),HEAPU32)[(buf+8)/4]=stats.bsize;(growMemViews(),HEAPU32)[(buf+72)/4]=stats.bsize;(growMemViews(),HEAP64)[(buf+16)/8]=BigInt(stats.blocks);(growMemViews(),HEAP64)[(buf+24)/8]=BigInt(stats.bfree);(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stats.bavail);(growMemViews(),HEAP64)[(buf+40)/8]=BigInt(stats.files);(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(stats.ffree);(growMemViews(),HEAPU32)[(buf+56)/4]=stats.fsid;(growMemViews(),HEAPU32)[(buf+80)/4]=stats.flags;(growMemViews(),HEAPU32)[(buf+64)/4]=stats.namelen},doMsync(addr,stream,len,flags,offset){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}if(flags&2){return 0}var buffer=(growMemViews(),HEAPU8).slice(addr,addr+len);FS.msync(stream,buffer,offset,len,flags)},getStreamFromFD(fd){var stream=FS.getStreamChecked(fd);return stream},varargs:undefined,getStr(ptr){var ret=UTF8ToString(ptr);return ret}};function ___syscall_fcntl64(fd,cmd,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(3,0,1,fd,cmd,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(cmd){case 0:{var arg=syscallGetVarargI();if(arg<0){return-28}while(FS.streams[arg]){arg++}var newStream;newStream=FS.dupStream(stream,arg);return newStream.fd}case 1:case 2:return 0;case 3:return stream.flags;case 4:{var arg=syscallGetVarargI();stream.flags|=arg;return 0}case 5:{var arg=syscallGetVarargP();var offset=0;(growMemViews(),HEAP16)[(arg+offset)/2]=2;return 0}case 6:case 7:return 0}return-28}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_fstat64(fd,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(4,0,1,fd,buf);buf=bigintToI53Checked(buf);try{return SYSCALLS.writeStat(buf,FS.fstat(fd))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var stringToUTF8=(str,outPtr,maxBytesToWrite)=>stringToUTF8Array(str,(growMemViews(),HEAPU8),outPtr,maxBytesToWrite);function ___syscall_getcwd(buf,size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(5,0,1,buf,size);buf=bigintToI53Checked(buf);size=bigintToI53Checked(size);try{if(size===0)return-28;var cwd=FS.cwd();var cwdLengthInBytes=lengthBytesUTF8(cwd)+1;if(size<cwdLengthInBytes)return-68;stringToUTF8(cwd,buf,size);return cwdLengthInBytes}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_getdents64(fd,dirp,count){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(6,0,1,fd,dirp,count);dirp=bigintToI53Checked(dirp);count=bigintToI53Checked(count);try{var stream=SYSCALLS.getStreamFromFD(fd);stream.getdents||=FS.readdir(stream.path);var struct_size=280;var pos=0;var off=FS.llseek(stream,0,1);var startIdx=Math.floor(off/struct_size);var endIdx=Math.min(stream.getdents.length,startIdx+Math.floor(count/struct_size));for(var idx=startIdx;idx<endIdx;idx++){var id;var type;var name=stream.getdents[idx];if(name==="."){id=stream.node.id;type=4}else if(name===".."){var lookup=FS.lookupPath(stream.path,{parent:true});id=lookup.node.id;type=4}else{var child;try{child=FS.lookupNode(stream.node,name)}catch(e){if(e?.errno===28){continue}throw e}id=child.id;type=FS.isChrdev(child.mode)?2:FS.isDir(child.mode)?4:FS.isLink(child.mode)?10:8}(growMemViews(),HEAP64)[(dirp+pos)/8]=BigInt(id);(growMemViews(),HEAP64)[(dirp+pos+8)/8]=BigInt((idx+1)*struct_size);(growMemViews(),HEAP16)[(dirp+pos+16)/2]=280;(growMemViews(),HEAP8)[dirp+pos+18]=type;stringToUTF8(name,dirp+pos+19,256);pos+=struct_size}FS.llseek(stream,idx*struct_size,0);return pos}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_ioctl(fd,op,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(7,0,1,fd,op,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(op){case 21509:{if(!stream.tty)return-59;return 0}case 21505:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcgets){var termios=stream.tty.ops.ioctl_tcgets(stream);var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=termios.c_iflag||0;(growMemViews(),HEAP32)[(argp+4)/4]=termios.c_oflag||0;(growMemViews(),HEAP32)[(argp+8)/4]=termios.c_cflag||0;(growMemViews(),HEAP32)[(argp+12)/4]=termios.c_lflag||0;for(var i=0;i<32;i++){(growMemViews(),HEAP8)[argp+i+17]=termios.c_cc[i]||0}return 0}return 0}case 21510:case 21511:case 21512:{if(!stream.tty)return-59;return 0}case 21506:case 21507:case 21508:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcsets){var argp=syscallGetVarargP();var c_iflag=(growMemViews(),HEAP32)[argp/4];var c_oflag=(growMemViews(),HEAP32)[(argp+4)/4];var c_cflag=(growMemViews(),HEAP32)[(argp+8)/4];var c_lflag=(growMemViews(),HEAP32)[(argp+12)/4];var c_cc=[];for(var i=0;i<32;i++){c_cc.push((growMemViews(),HEAP8)[argp+i+17])}return stream.tty.ops.ioctl_tcsets(stream.tty,op,{c_iflag,c_oflag,c_cflag,c_lflag,c_cc})}return 0}case 21519:{if(!stream.tty)return-59;var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=0;return 0}case 21520:{if(!stream.tty)return-59;return-28}case 21537:case 21531:{var argp=syscallGetVarargP();return FS.ioctl(stream,op,argp)}case 21523:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tiocgwinsz){var winsize=stream.tty.ops.ioctl_tiocgwinsz(stream.tty);var argp=syscallGetVarargP();(growMemViews(),HEAP16)[argp/2]=winsize[0];(growMemViews(),HEAP16)[(argp+2)/2]=winsize[1]}return 0}case 21524:{if(!stream.tty)return-59;return 0}case 21515:{if(!stream.tty)return-59;return 0}default:return-28}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_lstat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(8,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.lstat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_newfstatat(dirfd,path,buf,flags){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(9,0,1,dirfd,path,buf,flags);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);var nofollow=flags&256;var allowEmpty=flags&4096;flags=flags&~6400;path=SYSCALLS.calculateAt(dirfd,path,allowEmpty);return SYSCALLS.writeStat(buf,nofollow?FS.lstat(path):FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_openat(dirfd,path,flags,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(10,0,1,dirfd,path,flags,varargs);path=bigintToI53Checked(path);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{path=SYSCALLS.getStr(path);path=SYSCALLS.calculateAt(dirfd,path);var mode=varargs?syscallGetVarargI():0;return FS.open(path,flags,mode).fd}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_stat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(11,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __abort_js=()=>abort("");function __emscripten_init_main_thread_js(tb){tb=bigintToI53Checked(tb);__emscripten_thread_init(tb,!ENVIRONMENT_IS_WORKER,1,!ENVIRONMENT_IS_WEB,5242880,false);PThread.threadInitTLS()}var handleException=e=>{if(e instanceof ExitStatus||e=="unwind"){return EXITSTATUS}quit_(1,e)};var maybeExit=()=>{if(!keepRuntimeAlive()){try{if(ENVIRONMENT_IS_PTHREAD){if(_pthread_self())__emscripten_thread_exit(EXITSTATUS);return}_exit(EXITSTATUS)}catch(e){handleException(e)}}};var callUserCallback=func=>{if(ABORT){return}try{func();maybeExit()}catch(e){handleException(e)}};function __emscripten_thread_mailbox_await(pthread_ptr){pthread_ptr=bigintToI53Checked(pthread_ptr);if(Atomics.waitAsync){var wait=Atomics.waitAsync((growMemViews(),HEAP32),pthread_ptr/4,pthread_ptr);wait.value.then(checkMailbox);var waitingAsync=pthread_ptr+228;Atomics.store((growMemViews(),HEAP32),waitingAsync/4,1)}}var checkMailbox=()=>callUserCallback(()=>{var pthread_ptr=_pthread_self();if(pthread_ptr){__emscripten_thread_mailbox_await(pthread_ptr);__emscripten_check_mailbox()}});function __emscripten_notify_mailbox_postmessage(targetThread,currThreadId){targetThread=bigintToI53Checked(targetThread);currThreadId=bigintToI53Checked(currThreadId);if(targetThread==currThreadId){setTimeout(checkMailbox)}else if(ENVIRONMENT_IS_PTHREAD){postMessage({targetThread,cmd:"checkMailbox"})}else{var worker=PThread.pthreads[targetThread];if(!worker){return}worker.postMessage({cmd:"checkMailbox"})}}var proxiedJSCallArgs=[];function __emscripten_receive_on_main_thread_js(funcIndex,emAsmAddr,callingThread,numCallArgs,args){emAsmAddr=bigintToI53Checked(emAsmAddr);callingThread=bigintToI53Checked(callingThread);args=bigintToI53Checked(args);numCallArgs/=2;proxiedJSCallArgs.length=numCallArgs;var b=args/8;for(var i=0;i<numCallArgs;i++){if((growMemViews(),HEAP64)[b+2*i]){proxiedJSCallArgs[i]=(growMemViews(),HEAP64)[b+2*i+1]}else{proxiedJSCallArgs[i]=(growMemViews(),HEAPF64)[b+2*i+1]}}var func=proxiedFunctionTable[funcIndex];PThread.currentProxiedOperationCallerThread=callingThread;var rtn=func(...proxiedJSCallArgs);PThread.currentProxiedOperationCallerThread=0;if(typeof rtn=="bigint"){rtn=bigintToI53Checked(rtn)}return rtn}function __emscripten_thread_cleanup(thread){thread=bigintToI53Checked(thread);if(!ENVIRONMENT_IS_PTHREAD)cleanupThread(thread);else postMessage({cmd:"cleanupThread",thread})}function __emscripten_thread_set_strongref(thread){thread=bigintToI53Checked(thread);if(ENVIRONMENT_IS_NODE){PThread.pthreads[thread].ref()}}var isLeapYear=year=>year%4===0&&(year%100!==0||year%400===0);var MONTH_DAYS_LEAP_CUMULATIVE=[0,31,60,91,121,152,182,213,244,274,305,335];var MONTH_DAYS_REGULAR_CUMULATIVE=[0,31,59,90,120,151,181,212,243,273,304,334];var ydayFromDate=date=>{var leap=isLeapYear(date.getFullYear());var monthDaysCumulative=leap?MONTH_DAYS_LEAP_CUMULATIVE:MONTH_DAYS_REGULAR_CUMULATIVE;var yday=monthDaysCumulative[date.getMonth()]+date.getDate()-1;return yday};function __localtime_js(time,tmPtr){time=bigintToI53Checked(time);tmPtr=bigintToI53Checked(tmPtr);var date=new Date(time*1e3);(growMemViews(),HEAP32)[tmPtr/4]=date.getSeconds();(growMemViews(),HEAP32)[(tmPtr+4)/4]=date.getMinutes();(growMemViews(),HEAP32)[(tmPtr+8)/4]=date.getHours();(growMemViews(),HEAP32)[(tmPtr+12)/4]=date.getDate();(growMemViews(),HEAP32)[(tmPtr+16)/4]=date.getMonth();(growMemViews(),HEAP32)[(tmPtr+20)/4]=date.getFullYear()-1900;(growMemViews(),HEAP32)[(tmPtr+24)/4]=date.getDay();var yday=ydayFromDate(date)|0;(growMemViews(),HEAP32)[(tmPtr+28)/4]=yday;(growMemViews(),HEAP64)[(tmPtr+40)/8]=BigInt(-(date.getTimezoneOffset()*60));var start=new Date(date.getFullYear(),0,1);var summerOffset=new Date(date.getFullYear(),6,1).getTimezoneOffset();var winterOffset=start.getTimezoneOffset();var dst=(summerOffset!=winterOffset&&date.getTimezoneOffset()==Math.min(winterOffset,summerOffset))|0;(growMemViews(),HEAP32)[(tmPtr+32)/4]=dst}function __mmap_js(len,prot,flags,fd,offset,allocated,addr){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(12,0,1,len,prot,flags,fd,offset,allocated,addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);allocated=bigintToI53Checked(allocated);addr=bigintToI53Checked(addr);try{var stream=SYSCALLS.getStreamFromFD(fd);var res=FS.mmap(stream,len,offset,prot,flags);var ptr=res.ptr;(growMemViews(),HEAP32)[allocated/4]=res.allocated;(growMemViews(),HEAPU64)[addr/8]=BigInt(ptr);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function __munmap_js(addr,len,prot,flags,fd,offset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(13,0,1,addr,len,prot,flags,fd,offset);addr=bigintToI53Checked(addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);try{var stream=SYSCALLS.getStreamFromFD(fd);if(prot&2){SYSCALLS.doMsync(addr,stream,len,flags,offset)}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __tzset_js=function(timezone,daylight,std_name,dst_name){timezone=bigintToI53Checked(timezone);daylight=bigintToI53Checked(daylight);std_name=bigintToI53Checked(std_name);dst_name=bigintToI53Checked(dst_name);var currentYear=(new Date).getFullYear();var winter=new Date(currentYear,0,1);var summer=new Date(currentYear,6,1);var winterOffset=winter.getTimezoneOffset();var summerOffset=summer.getTimezoneOffset();var stdTimezoneOffset=Math.max(winterOffset,summerOffset);(growMemViews(),HEAPU64)[timezone/8]=BigInt(stdTimezoneOffset*60);(growMemViews(),HEAP32)[daylight/4]=Number(winterOffset!=summerOffset);var extractZone=timezoneOffset=>{var sign=timezoneOffset>=0?"-":"+";var absOffset=Math.abs(timezoneOffset);var hours=String(Math.floor(absOffset/60)).padStart(2,"0");var minutes=String(absOffset%60).padStart(2,"0");return`UTC${sign}${hours}${minutes}`};var winterName=extractZone(winterOffset);var summerName=extractZone(summerOffset);if(summerOffset<winterOffset){stringToUTF8(winterName,std_name,17);stringToUTF8(summerName,dst_name,17)}else{stringToUTF8(winterName,dst_name,17);stringToUTF8(summerName,std_name,17)}};var _emscripten_get_now=()=>performance.timeOrigin+performance.now();var _emscripten_date_now=()=>Date.now();var nowIsMonotonic=1;var checkWasiClock=clock_id=>clock_id>=0&&clock_id<=3;function _clock_time_get(clk_id,ignored_precision,ptime){ignored_precision=bigintToI53Checked(ignored_precision);ptime=bigintToI53Checked(ptime);if(!checkWasiClock(clk_id)){return 28}var now;if(clk_id===0){now=_emscripten_date_now()}else if(nowIsMonotonic){now=_emscripten_get_now()}else{return 52}var nsec=Math.round(now*1e3*1e3);(growMemViews(),HEAP64)[ptime/8]=BigInt(nsec);return 0}var _emscripten_check_blocking_allowed=()=>{};var runtimeKeepalivePush=()=>{runtimeKeepaliveCounter+=1};var _emscripten_exit_with_live_runtime=()=>{runtimeKeepalivePush();throw"unwind"};var jsStackTrace=()=>(new Error).stack.toString();var getCallstack=flags=>{var callstack=jsStackTrace();var lines=callstack.split("\\n");callstack="";var firefoxRe=new RegExp("\\\\s*(.*?)@(.*?):([0-9]+):([0-9]+)");var chromeRe=new RegExp("\\\\s*at (.*?) \\\\((.*):(.*):(.*)\\\\)");for(var line of lines){var symbolName="";var file="";var lineno=0;var column=0;var parts=chromeRe.exec(line);if(parts?.length==5){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]}else{parts=firefoxRe.exec(line);if(parts?.length>=4){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]|0}else{callstack+=line+"\\n";continue}}if(symbolName=="_emscripten_log"||symbolName=="_emscripten_get_callstack"){callstack="";continue}if(flags&24){if(flags&64){file=file.substring(file.replace(/\\\\/g,"/").lastIndexOf("/")+1)}callstack+=`    at ${symbolName} (${file}:${lineno}:${column})\\n`}}callstack=callstack.replace(/\\s+$/,"");return callstack};function _emscripten_get_callstack(flags,str,maxbytes){str=bigintToI53Checked(str);var callstack=getCallstack(flags);if(!str||maxbytes<=0){return lengthBytesUTF8(callstack)+1}var bytesWrittenExcludingNull=stringToUTF8(callstack,str,maxbytes);return bytesWrittenExcludingNull+1}var getHeapMax=()=>4294967296;var _emscripten_get_heap_max=()=>BigInt(getHeapMax());var _emscripten_has_asyncify=()=>2;var _emscripten_num_logical_cores=()=>ENVIRONMENT_IS_NODE?require("os").cpus().length:navigator["hardwareConcurrency"];var growMemory=size=>{var oldHeapSize=wasmMemory.buffer.byteLength;var pages=(size-oldHeapSize+65535)/65536|0;try{wasmMemory.grow(BigInt(pages));updateMemoryViews();return 1}catch(e){}};function _emscripten_resize_heap(requestedSize){requestedSize=bigintToI53Checked(requestedSize);var oldSize=(growMemViews(),HEAPU8).length;if(requestedSize<=oldSize){return false}var maxHeapSize=getHeapMax();if(requestedSize>maxHeapSize){return false}for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignMemory(Math.max(requestedSize,overGrownHeapSize),65536));var replacement=growMemory(newSize);if(replacement){return true}}return false}var stringToUTF8OnStack=str=>{var size=lengthBytesUTF8(str)+1;var ret=stackAlloc(size);stringToUTF8(str,ret,size);return ret};var writeI53ToI64=(ptr,num)=>{(growMemViews(),HEAPU32)[ptr/4]=num;var lower=(growMemViews(),HEAPU32)[ptr/4];(growMemViews(),HEAPU32)[(ptr+4)/4]=(num-lower)/4294967296};var stringToNewUTF8=str=>{var size=lengthBytesUTF8(str)+1;var ret=_malloc(size);if(ret)stringToUTF8(str,ret,size);return ret};var readI53FromI64=ptr=>(growMemViews(),HEAPU32)[ptr/4]+(growMemViews(),HEAP32)[(ptr+4)/4]*4294967296;var WebGPU={Internals:{jsObjects:[],jsObjectInsert:(ptr,jsObject)=>{WebGPU.Internals.jsObjects[ptr]=jsObject},bufferOnUnmaps:[],futures:[],futureInsert:(futureId,promise)=>{WebGPU.Internals.futures[futureId]=new Promise(resolve=>promise.finally(()=>resolve(futureId)))}},getJsObject:ptr=>{if(!ptr)return undefined;return WebGPU.Internals.jsObjects[ptr]},importJsAdapter:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateAdapter(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroup:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroup(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroupLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroupLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBuffer:(buffer,parentPtr=0)=>{assert(buffer.mapState==="unmapped");var bufferPtr=_emwgpuCreateBuffer(parentPtr);WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);return bufferPtr},importJsCommandBuffer:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandBuffer(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsCommandEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsDevice:(device,parentPtr=0)=>{var queuePtr=_emwgpuCreateQueue(parentPtr);var devicePtr=_emwgpuCreateDevice(parentPtr,queuePtr);WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);return devicePtr},importJsExternalTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateExternalTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsPipelineLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreatePipelineLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQuerySet:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQuerySet(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQueue:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQueue(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundle:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundle(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundleEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundleEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSampler:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSampler(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsShaderModule:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateShaderModule(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSurface:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSurface(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTextureView:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTextureView(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},errorCallback:(callback,type,message,userdata)=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(message);((a1,a2,a3)=>getWasmTableEntry(callback).call(null,a1,BigInt(a2),BigInt(a3)))(type,BigInt(messagePtr),userdata);stackRestore(sp)},iterateExtensions:(root,handlers)=>{for(var ptr=Number((growMemViews(),HEAPU64)[root/8]);ptr;ptr=Number((growMemViews(),HEAPU64)[ptr/8])){var sType=(growMemViews(),HEAP32)[(ptr+8)/4];var handler=handlers[sType](ptr)}},setStringView:(ptr,data,length)=>{(growMemViews(),HEAPU64)[ptr/8]=BigInt(data);(growMemViews(),HEAPU64)[(ptr+8)/8]=BigInt(length)},makeStringFromStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);return UTF8ToString(ptr,length)},makeStringFromOptionalStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);if(!ptr){if(length===0){return""}return undefined}return UTF8ToString(ptr,length)},makeColor:ptr=>({r:(growMemViews(),HEAPF64)[ptr/8],g:(growMemViews(),HEAPF64)[(ptr+8)/8],b:(growMemViews(),HEAPF64)[(ptr+16)/8],a:(growMemViews(),HEAPF64)[(ptr+24)/8]}),makeExtent3D:ptr=>({width:(growMemViews(),HEAPU32)[ptr/4],height:(growMemViews(),HEAPU32)[(ptr+4)/4],depthOrArrayLayers:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeOrigin3D:ptr=>({x:(growMemViews(),HEAPU32)[ptr/4],y:(growMemViews(),HEAPU32)[(ptr+4)/4],z:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeTexelCopyTextureInfo:ptr=>({texture:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[ptr/8])),mipLevel:(growMemViews(),HEAPU32)[(ptr+8)/4],origin:WebGPU.makeOrigin3D(ptr+12),aspect:WebGPU.TextureAspect[(growMemViews(),HEAP32)[(ptr+24)/4]]}),makeTexelCopyBufferLayout:ptr=>{var bytesPerRow=(growMemViews(),HEAPU32)[(ptr+8)/4];var rowsPerImage=(growMemViews(),HEAPU32)[(ptr+12)/4];return{offset:readI53FromI64(ptr),bytesPerRow:bytesPerRow===4294967295?undefined:bytesPerRow,rowsPerImage:rowsPerImage===4294967295?undefined:rowsPerImage}},makeTexelCopyBufferInfo:ptr=>{var layoutPtr=ptr+0;var bufferCopyView=WebGPU.makeTexelCopyBufferLayout(layoutPtr);bufferCopyView["buffer"]=WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+16)/8]));return bufferCopyView},makePassTimestampWrites:ptr=>{if(ptr===0)return undefined;return{querySet:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),beginningOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+16)/4],endOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+20)/4]}},makePipelineConstants:(constantCount,constantsPtr)=>{if(!constantCount)return;var constants={};for(var i=0;i<constantCount;++i){var entryPtr=constantsPtr+32*i;var key=WebGPU.makeStringFromStringView(entryPtr+8);constants[key]=(growMemViews(),HEAPF64)[(entryPtr+24)/8]}return constants},makePipelineLayout:layoutPtr=>{if(!layoutPtr)return"auto";return WebGPU.getJsObject(layoutPtr)},makeComputeState:ptr=>{if(!ptr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(ptr+32)/8]),Number((growMemViews(),HEAPU64)[(ptr+40)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(ptr+16)};return desc},makeComputePipelineDesc:descriptor=>{var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),compute:WebGPU.makeComputeState(descriptor+32)};return desc},makeRenderPipelineDesc:descriptor=>{function makePrimitiveState(psPtr){if(!psPtr)return undefined;return{topology:WebGPU.PrimitiveTopology[(growMemViews(),HEAP32)[(psPtr+8)/4]],stripIndexFormat:WebGPU.IndexFormat[(growMemViews(),HEAP32)[(psPtr+12)/4]],frontFace:WebGPU.FrontFace[(growMemViews(),HEAP32)[(psPtr+16)/4]],cullMode:WebGPU.CullMode[(growMemViews(),HEAP32)[(psPtr+20)/4]],unclippedDepth:!!(growMemViews(),HEAPU32)[(psPtr+24)/4]}}function makeBlendComponent(bdPtr){if(!bdPtr)return undefined;return{operation:WebGPU.BlendOperation[(growMemViews(),HEAP32)[bdPtr/4]],srcFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+4)/4]],dstFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+8)/4]]}}function makeBlendState(bsPtr){if(!bsPtr)return undefined;return{alpha:makeBlendComponent(bsPtr+12),color:makeBlendComponent(bsPtr+0)}}function makeColorState(csPtr){var format=WebGPU.TextureFormat[(growMemViews(),HEAP32)[(csPtr+8)/4]];return format?{format,blend:makeBlendState(Number((growMemViews(),HEAPU64)[(csPtr+16)/8])),writeMask:(growMemViews(),HEAPU32)[(csPtr+24)/4]}:undefined}function makeColorStates(count,csArrayPtr){var states=[];for(var i=0;i<count;++i){states.push(makeColorState(csArrayPtr+32*i))}return states}function makeStencilStateFace(ssfPtr){return{compare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[ssfPtr/4]],failOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+4)/4]],depthFailOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+8)/4]],passOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+12)/4]]}}function makeDepthStencilState(dssPtr){if(!dssPtr)return undefined;return{format:WebGPU.TextureFormat[(growMemViews(),HEAP32)[(dssPtr+8)/4]],depthWriteEnabled:!!(growMemViews(),HEAPU32)[(dssPtr+12)/4],depthCompare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[(dssPtr+16)/4]],stencilFront:makeStencilStateFace(dssPtr+20),stencilBack:makeStencilStateFace(dssPtr+36),stencilReadMask:(growMemViews(),HEAPU32)[(dssPtr+52)/4],stencilWriteMask:(growMemViews(),HEAPU32)[(dssPtr+56)/4],depthBias:(growMemViews(),HEAP32)[(dssPtr+60)/4],depthBiasSlopeScale:(growMemViews(),HEAPF32)[(dssPtr+64)/4],depthBiasClamp:(growMemViews(),HEAPF32)[(dssPtr+68)/4]}}function makeVertexAttribute(vaPtr){return{format:WebGPU.VertexFormat[(growMemViews(),HEAP32)[(vaPtr+8)/4]],offset:readI53FromI64(vaPtr+16),shaderLocation:(growMemViews(),HEAPU32)[(vaPtr+24)/4]}}function makeVertexAttributes(count,vaArrayPtr){var vas=[];for(var i=0;i<count;++i){vas.push(makeVertexAttribute(vaArrayPtr+i*32))}return vas}function makeVertexBuffer(vbPtr){if(!vbPtr)return undefined;var stepMode=WebGPU.VertexStepMode[(growMemViews(),HEAP32)[(vbPtr+8)/4]];var attributeCount=Number((growMemViews(),HEAPU64)[(vbPtr+24)/8]);if(!stepMode&&!attributeCount){return null}return{arrayStride:readI53FromI64(vbPtr+16),stepMode,attributes:makeVertexAttributes(attributeCount,Number((growMemViews(),HEAPU64)[(vbPtr+32)/8]))}}function makeVertexBuffers(count,vbArrayPtr){if(!count)return undefined;var vbs=[];for(var i=0;i<count;++i){vbs.push(makeVertexBuffer(vbArrayPtr+i*40))}return vbs}function makeVertexState(viPtr){if(!viPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(viPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(viPtr+32)/8]),Number((growMemViews(),HEAPU64)[(viPtr+40)/8])),buffers:makeVertexBuffers(Number((growMemViews(),HEAPU64)[(viPtr+48)/8]),Number((growMemViews(),HEAPU64)[(viPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(viPtr+16)};return desc}function makeMultisampleState(msPtr){if(!msPtr)return undefined;return{count:(growMemViews(),HEAPU32)[(msPtr+8)/4],mask:(growMemViews(),HEAPU32)[(msPtr+12)/4],alphaToCoverageEnabled:!!(growMemViews(),HEAPU32)[(msPtr+16)/4]}}function makeFragmentState(fsPtr){if(!fsPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(fsPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(fsPtr+32)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+40)/8])),targets:makeColorStates(Number((growMemViews(),HEAPU64)[(fsPtr+48)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(fsPtr+16)};return desc}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),vertex:makeVertexState(descriptor+32),primitive:makePrimitiveState(descriptor+96),depthStencil:makeDepthStencilState(Number((growMemViews(),HEAPU64)[(descriptor+128)/8])),multisample:makeMultisampleState(descriptor+136),fragment:makeFragmentState(Number((growMemViews(),HEAPU64)[(descriptor+160)/8]))};return desc},fillLimitStruct:(limits,limitsOutPtr)=>{var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsOutPtr/8]);function setLimitValueU32(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;(growMemViews(),HEAPU32)[(basePtr+limitOffset)/4]=limitValue}function setLimitValueU64(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;writeI53ToI64(basePtr+limitOffset,limitValue)}setLimitValueU32("maxTextureDimension1D",limitsOutPtr,8);setLimitValueU32("maxTextureDimension2D",limitsOutPtr,12);setLimitValueU32("maxTextureDimension3D",limitsOutPtr,16);setLimitValueU32("maxTextureArrayLayers",limitsOutPtr,20);setLimitValueU32("maxBindGroups",limitsOutPtr,24);setLimitValueU32("maxBindGroupsPlusVertexBuffers",limitsOutPtr,28);setLimitValueU32("maxBindingsPerBindGroup",limitsOutPtr,32);setLimitValueU32("maxDynamicUniformBuffersPerPipelineLayout",limitsOutPtr,36);setLimitValueU32("maxDynamicStorageBuffersPerPipelineLayout",limitsOutPtr,40);setLimitValueU32("maxSampledTexturesPerShaderStage",limitsOutPtr,44);setLimitValueU32("maxSamplersPerShaderStage",limitsOutPtr,48);setLimitValueU32("maxStorageBuffersPerShaderStage",limitsOutPtr,52);setLimitValueU32("maxStorageTexturesPerShaderStage",limitsOutPtr,56);setLimitValueU32("maxUniformBuffersPerShaderStage",limitsOutPtr,60);setLimitValueU32("minUniformBufferOffsetAlignment",limitsOutPtr,80);setLimitValueU32("minStorageBufferOffsetAlignment",limitsOutPtr,84);setLimitValueU64("maxUniformBufferBindingSize",limitsOutPtr,64);setLimitValueU64("maxStorageBufferBindingSize",limitsOutPtr,72);setLimitValueU32("maxVertexBuffers",limitsOutPtr,88);setLimitValueU64("maxBufferSize",limitsOutPtr,96);setLimitValueU32("maxVertexAttributes",limitsOutPtr,104);setLimitValueU32("maxVertexBufferArrayStride",limitsOutPtr,108);setLimitValueU32("maxInterStageShaderVariables",limitsOutPtr,112);setLimitValueU32("maxColorAttachments",limitsOutPtr,116);setLimitValueU32("maxColorAttachmentBytesPerSample",limitsOutPtr,120);setLimitValueU32("maxComputeWorkgroupStorageSize",limitsOutPtr,124);setLimitValueU32("maxComputeInvocationsPerWorkgroup",limitsOutPtr,128);setLimitValueU32("maxComputeWorkgroupSizeX",limitsOutPtr,132);setLimitValueU32("maxComputeWorkgroupSizeY",limitsOutPtr,136);setLimitValueU32("maxComputeWorkgroupSizeZ",limitsOutPtr,140);setLimitValueU32("maxComputeWorkgroupsPerDimension",limitsOutPtr,144);setLimitValueU32("maxImmediateSize",limitsOutPtr,148);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;setLimitValueU32("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20,limits.maxStorageTexturesPerShaderStage);setLimitValueU32("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28,limits.maxStorageTexturesPerShaderStage)}},fillAdapterInfoStruct:(info,infoStruct)=>{(growMemViews(),HEAPU32)[(infoStruct+88)/4]=info.subgroupMinSize;(growMemViews(),HEAPU32)[(infoStruct+92)/4]=info.subgroupMaxSize;var strs=info.vendor+info.architecture+info.device+info.description;var strPtr=stringToNewUTF8(strs);var vendorLen=lengthBytesUTF8(info.vendor);WebGPU.setStringView(infoStruct+8,strPtr,vendorLen);strPtr+=vendorLen;var architectureLen=lengthBytesUTF8(info.architecture);WebGPU.setStringView(infoStruct+24,strPtr,architectureLen);strPtr+=architectureLen;var deviceLen=lengthBytesUTF8(info.device);WebGPU.setStringView(infoStruct+40,strPtr,deviceLen);strPtr+=deviceLen;var descriptionLen=lengthBytesUTF8(info.description);WebGPU.setStringView(infoStruct+56,strPtr,descriptionLen);strPtr+=descriptionLen;(growMemViews(),HEAP32)[(infoStruct+72)/4]=2;var adapterType=info.isFallbackAdapter?3:4;(growMemViews(),HEAP32)[(infoStruct+76)/4]=adapterType;(growMemViews(),HEAPU32)[(infoStruct+80)/4]=0;(growMemViews(),HEAPU32)[(infoStruct+84)/4]=0},AddressMode:[,"clamp-to-edge","repeat","mirror-repeat"],BlendFactor:[,"zero","one","src","one-minus-src","src-alpha","one-minus-src-alpha","dst","one-minus-dst","dst-alpha","one-minus-dst-alpha","src-alpha-saturated","constant","one-minus-constant","src1","one-minus-src1","src1-alpha","one-minus-src1-alpha"],BlendOperation:[,"add","subtract","reverse-subtract","min","max"],BufferBindingType:[,,"uniform","storage","read-only-storage"],BufferMapState:[,"unmapped","pending","mapped"],CompareFunction:[,"never","less","equal","less-equal","greater","not-equal","greater-equal","always"],CompilationInfoRequestStatus:[,"success","callback-cancelled"],ComponentSwizzle:[,"0","1","r","g","b","a"],CompositeAlphaMode:[,"opaque","premultiplied","unpremultiplied","inherit"],CullMode:[,"none","front","back"],ErrorFilter:[,"validation","out-of-memory","internal"],FeatureLevel:[,"compatibility","core"],FeatureName:{1:"core-features-and-limits",2:"depth-clip-control",3:"depth32float-stencil8",4:"texture-compression-bc",5:"texture-compression-bc-sliced-3d",6:"texture-compression-etc2",7:"texture-compression-astc",8:"texture-compression-astc-sliced-3d",9:"timestamp-query",10:"indirect-first-instance",11:"shader-f16",12:"rg11b10ufloat-renderable",13:"bgra8unorm-storage",14:"float32-filterable",15:"float32-blendable",16:"clip-distances",17:"dual-source-blending",18:"subgroups",19:"texture-formats-tier1",20:"texture-formats-tier2",21:"primitive-index",22:"texture-component-swizzle",327692:"chromium-experimental-unorm16-texture-formats",327729:"chromium-experimental-multi-draw-indirect"},FilterMode:[,"nearest","linear"],FrontFace:[,"ccw","cw"],IndexFormat:[,"uint16","uint32"],InstanceFeatureName:[,"timed-wait-any","shader-source-spirv","multiple-devices-per-adapter"],LoadOp:[,"load","clear"],MipmapFilterMode:[,"nearest","linear"],OptionalBool:["false","true"],PowerPreference:[,"low-power","high-performance"],PredefinedColorSpace:[,"srgb","display-p3"],PrimitiveTopology:[,"point-list","line-list","line-strip","triangle-list","triangle-strip"],QueryType:[,"occlusion","timestamp"],SamplerBindingType:[,,"filtering","non-filtering","comparison"],Status:[,"success","error"],StencilOperation:[,"keep","zero","replace","invert","increment-clamp","decrement-clamp","increment-wrap","decrement-wrap"],StorageTextureAccess:[,,"write-only","read-only","read-write"],StoreOp:[,"store","discard"],SurfaceGetCurrentTextureStatus:[,"success-optimal","success-suboptimal","timeout","outdated","lost","error"],TextureAspect:[,"all","stencil-only","depth-only"],TextureDimension:[,"1d","2d","3d"],TextureFormat:[,"r8unorm","r8snorm","r8uint","r8sint","r16unorm","r16snorm","r16uint","r16sint","r16float","rg8unorm","rg8snorm","rg8uint","rg8sint","r32float","r32uint","r32sint","rg16unorm","rg16snorm","rg16uint","rg16sint","rg16float","rgba8unorm","rgba8unorm-srgb","rgba8snorm","rgba8uint","rgba8sint","bgra8unorm","bgra8unorm-srgb","rgb10a2uint","rgb10a2unorm","rg11b10ufloat","rgb9e5ufloat","rg32float","rg32uint","rg32sint","rgba16unorm","rgba16snorm","rgba16uint","rgba16sint","rgba16float","rgba32float","rgba32uint","rgba32sint","stencil8","depth16unorm","depth24plus","depth24plus-stencil8","depth32float","depth32float-stencil8","bc1-rgba-unorm","bc1-rgba-unorm-srgb","bc2-rgba-unorm","bc2-rgba-unorm-srgb","bc3-rgba-unorm","bc3-rgba-unorm-srgb","bc4-r-unorm","bc4-r-snorm","bc5-rg-unorm","bc5-rg-snorm","bc6h-rgb-ufloat","bc6h-rgb-float","bc7-rgba-unorm","bc7-rgba-unorm-srgb","etc2-rgb8unorm","etc2-rgb8unorm-srgb","etc2-rgb8a1unorm","etc2-rgb8a1unorm-srgb","etc2-rgba8unorm","etc2-rgba8unorm-srgb","eac-r11unorm","eac-r11snorm","eac-rg11unorm","eac-rg11snorm","astc-4x4-unorm","astc-4x4-unorm-srgb","astc-5x4-unorm","astc-5x4-unorm-srgb","astc-5x5-unorm","astc-5x5-unorm-srgb","astc-6x5-unorm","astc-6x5-unorm-srgb","astc-6x6-unorm","astc-6x6-unorm-srgb","astc-8x5-unorm","astc-8x5-unorm-srgb","astc-8x6-unorm","astc-8x6-unorm-srgb","astc-8x8-unorm","astc-8x8-unorm-srgb","astc-10x5-unorm","astc-10x5-unorm-srgb","astc-10x6-unorm","astc-10x6-unorm-srgb","astc-10x8-unorm","astc-10x8-unorm-srgb","astc-10x10-unorm","astc-10x10-unorm-srgb","astc-12x10-unorm","astc-12x10-unorm-srgb","astc-12x12-unorm","astc-12x12-unorm-srgb"],TextureSampleType:[,,"float","unfilterable-float","depth","sint","uint"],TextureViewDimension:[,"1d","2d","2d-array","cube","cube-array","3d"],ToneMappingMode:[,"standard","extended"],VertexFormat:[,"uint8","uint8x2","uint8x4","sint8","sint8x2","sint8x4","unorm8","unorm8x2","unorm8x4","snorm8","snorm8x2","snorm8x4","uint16","uint16x2","uint16x4","sint16","sint16x2","sint16x4","unorm16","unorm16x2","unorm16x4","snorm16","snorm16x2","snorm16x4","float16","float16x2","float16x4","float32","float32x2","float32x3","float32x4","uint32","uint32x2","uint32x3","uint32x4","sint32","sint32x2","sint32x3","sint32x4","unorm10-10-10-2","unorm8x4-bgra"],VertexStepMode:[,"vertex","instance"],WGSLLanguageFeatureName:[,"readonly_and_readwrite_storage_textures","packed_4x8_integer_dot_product","unrestricted_pointer_parameters","pointer_composite_access","uniform_buffer_standard_layout","subgroup_id","texture_and_sampler_let","subgroup_uniformity","texture_formats_tier1"]};var emwgpuStringToInt_DeviceLostReason={undefined:1,unknown:1,destroyed:2};var runtimeKeepalivePop=()=>{runtimeKeepaliveCounter-=1};function _emwgpuAdapterRequestDevice(adapterPtr,futureId,deviceLostFutureId,devicePtr,queuePtr,descriptor){adapterPtr=bigintToI53Checked(adapterPtr);futureId=bigintToI53Checked(futureId);deviceLostFutureId=bigintToI53Checked(deviceLostFutureId);devicePtr=bigintToI53Checked(devicePtr);queuePtr=bigintToI53Checked(queuePtr);descriptor=bigintToI53Checked(descriptor);var adapter=WebGPU.getJsObject(adapterPtr);var desc={};if(descriptor){var requiredFeatureCount=Number((growMemViews(),HEAPU64)[(descriptor+24)/8]);if(requiredFeatureCount){var requiredFeaturesPtr=Number((growMemViews(),HEAPU64)[(descriptor+32)/8]);desc["requiredFeatures"]=Array.from((growMemViews(),HEAPU32).subarray(requiredFeaturesPtr/4,(requiredFeaturesPtr+requiredFeatureCount*4)/4),feature=>WebGPU.FeatureName[feature])}var limitsPtr=Number((growMemViews(),HEAPU64)[(descriptor+40)/8]);if(limitsPtr){var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsPtr/8]);var requiredLimits={};function setLimitU32IfDefined(name,basePtr,limitOffset,ignoreIfZero=false){var ptr=basePtr+limitOffset;var value=(growMemViews(),HEAPU32)[ptr/4];if(value!=4294967295&&(!ignoreIfZero||value!=0)){requiredLimits[name]=value}}function setLimitU64IfDefined(name,basePtr,limitOffset){var ptr=basePtr+limitOffset;var limitPart1=(growMemViews(),HEAPU32)[ptr/4];var limitPart2=(growMemViews(),HEAPU32)[(ptr+4)/4];if(limitPart1!=4294967295||limitPart2!=4294967295){requiredLimits[name]=readI53FromI64(ptr)}}setLimitU32IfDefined("maxTextureDimension1D",limitsPtr,8);setLimitU32IfDefined("maxTextureDimension2D",limitsPtr,12);setLimitU32IfDefined("maxTextureDimension3D",limitsPtr,16);setLimitU32IfDefined("maxTextureArrayLayers",limitsPtr,20);setLimitU32IfDefined("maxBindGroups",limitsPtr,24);setLimitU32IfDefined("maxBindGroupsPlusVertexBuffers",limitsPtr,28);setLimitU32IfDefined("maxBindingsPerBindGroup",limitsPtr,32);setLimitU32IfDefined("maxDynamicUniformBuffersPerPipelineLayout",limitsPtr,36);setLimitU32IfDefined("maxDynamicStorageBuffersPerPipelineLayout",limitsPtr,40);setLimitU32IfDefined("maxSampledTexturesPerShaderStage",limitsPtr,44);setLimitU32IfDefined("maxSamplersPerShaderStage",limitsPtr,48);setLimitU32IfDefined("maxStorageBuffersPerShaderStage",limitsPtr,52);setLimitU32IfDefined("maxStorageTexturesPerShaderStage",limitsPtr,56);setLimitU32IfDefined("maxUniformBuffersPerShaderStage",limitsPtr,60);setLimitU32IfDefined("minUniformBufferOffsetAlignment",limitsPtr,80);setLimitU32IfDefined("minStorageBufferOffsetAlignment",limitsPtr,84);setLimitU64IfDefined("maxUniformBufferBindingSize",limitsPtr,64);setLimitU64IfDefined("maxStorageBufferBindingSize",limitsPtr,72);setLimitU32IfDefined("maxVertexBuffers",limitsPtr,88);setLimitU64IfDefined("maxBufferSize",limitsPtr,96);setLimitU32IfDefined("maxVertexAttributes",limitsPtr,104);setLimitU32IfDefined("maxVertexBufferArrayStride",limitsPtr,108);setLimitU32IfDefined("maxInterStageShaderVariables",limitsPtr,112);setLimitU32IfDefined("maxColorAttachments",limitsPtr,116);setLimitU32IfDefined("maxColorAttachmentBytesPerSample",limitsPtr,120);setLimitU32IfDefined("maxComputeWorkgroupStorageSize",limitsPtr,124);setLimitU32IfDefined("maxComputeInvocationsPerWorkgroup",limitsPtr,128);setLimitU32IfDefined("maxComputeWorkgroupSizeX",limitsPtr,132);setLimitU32IfDefined("maxComputeWorkgroupSizeY",limitsPtr,136);setLimitU32IfDefined("maxComputeWorkgroupSizeZ",limitsPtr,140);setLimitU32IfDefined("maxComputeWorkgroupsPerDimension",limitsPtr,144);setLimitU32IfDefined("maxImmediateSize",limitsPtr,148,true);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;if("maxStorageBuffersInVertexStage"in GPUSupportedLimits.prototype){setLimitU32IfDefined("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16);setLimitU32IfDefined("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20);setLimitU32IfDefined("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24);setLimitU32IfDefined("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28)}}desc["requiredLimits"]=requiredLimits}var defaultQueuePtr=Number((growMemViews(),HEAPU64)[(descriptor+48)/8]);if(defaultQueuePtr){var defaultQueueDesc={label:WebGPU.makeStringFromOptionalStringView(defaultQueuePtr+8)};desc["defaultQueue"]=defaultQueueDesc}desc["label"]=WebGPU.makeStringFromOptionalStringView(descriptor+8)}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,adapter.requestDevice(desc).then(device=>{runtimeKeepalivePop();callUserCallback(()=>{WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);devicePtr=BigInt(devicePtr);WebGPU.Internals.futureInsert(deviceLostFutureId,device.lost.then(info=>{callUserCallback(()=>{device.onuncapturederror=ev=>{};var sp=stackSave();var messagePtr=stringToUTF8OnStack(info.message);_emwgpuOnDeviceLostCompleted(deviceLostFutureId,emwgpuStringToInt_DeviceLostReason[info.reason],BigInt(messagePtr));stackRestore(sp)})}));device.onuncapturederror=ev=>{var type=5;if(ev.error instanceof GPUValidationError)type=2;else if(ev.error instanceof GPUOutOfMemoryError)type=3;else if(ev.error instanceof GPUInternalError)type=4;var sp=stackSave();var messagePtr=stringToUTF8OnStack(ev.error.message);_emwgpuOnUncapturedError(BigInt(devicePtr),type,BigInt(messagePtr));stackRestore(sp)};_emwgpuOnRequestDeviceCompleted(futureId,1,BigInt(devicePtr),0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestDeviceCompleted(futureId,3,BigInt(devicePtr),BigInt(messagePtr));if(deviceLostFutureId){_emwgpuOnDeviceLostCompleted(deviceLostFutureId,4,BigInt(messagePtr))}stackRestore(sp)})}))}function _emwgpuBufferDestroy(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(onUnmap){for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]}buffer.destroy()}var warnOnce=text=>{warnOnce.shown||={};if(!warnOnce.shown[text]){warnOnce.shown[text]=1;if(ENVIRONMENT_IS_NODE)text="warning: "+text;err(text)}};var _emwgpuBufferGetConstMappedRange=function(bufferPtr,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);if(size==-1)size=undefined;var mapped;try{mapped=buffer.getMappedRange(offset,size)}catch(ex){return 0n}var data=_memalign(16,mapped.byteLength);(growMemViews(),HEAPU8).set(new Uint8Array(mapped),data);WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(()=>_free(data));return data})();return BigInt(ret)};var _emwgpuBufferMapAsync=function(bufferPtr,futureId,mode,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);futureId=bigintToI53Checked(futureId);mode=bigintToI53Checked(mode);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var buffer=WebGPU.getJsObject(bufferPtr);WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[];if(size==-1)size=undefined;runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,buffer.mapAsync(mode,offset,size).then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnMapAsyncCompleted(futureId,1,0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);var status=ex.name==="AbortError"?4:ex.name==="OperationError"?3:0;_emwgpuOnMapAsyncCompleted(futureId,status,BigInt(messagePtr));delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]})}))};function _emwgpuBufferUnmap(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(!onUnmap){return}for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];buffer.unmap()}function _emwgpuDelete(ptr){ptr=bigintToI53Checked(ptr);delete WebGPU.Internals.jsObjects[ptr]}function _emwgpuDeviceCreateBuffer(devicePtr,descriptor,bufferPtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);bufferPtr=bigintToI53Checked(bufferPtr);var mappedAtCreation=!!(growMemViews(),HEAPU32)[(descriptor+40)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),usage:(growMemViews(),HEAPU32)[(descriptor+24)/4],size:readI53FromI64(descriptor+32),mappedAtCreation};var device=WebGPU.getJsObject(devicePtr);var buffer;try{buffer=device.createBuffer(desc)}catch(ex){return false}WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);if(mappedAtCreation){WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[]}return true}function _emwgpuDeviceCreateShaderModule(devicePtr,descriptor,shaderModulePtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);shaderModulePtr=bigintToI53Checked(shaderModulePtr);var nextInChainPtr=Number((growMemViews(),HEAPU64)[descriptor/8]);var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),code:""};switch(sType){case 2:{desc["code"]=WebGPU.makeStringFromStringView(nextInChainPtr+16);break}}var device=WebGPU.getJsObject(devicePtr);WebGPU.Internals.jsObjectInsert(shaderModulePtr,device.createShaderModule(desc))}var _emwgpuDeviceDestroy=devicePtr=>{const device=WebGPU.getJsObject(devicePtr);device.onuncapturederror=null;device.destroy()};function _emwgpuInstanceRequestAdapter(instancePtr,futureId,options,adapterPtr){instancePtr=bigintToI53Checked(instancePtr);futureId=bigintToI53Checked(futureId);options=bigintToI53Checked(options);adapterPtr=bigintToI53Checked(adapterPtr);var opts;if(options){opts={featureLevel:WebGPU.FeatureLevel[(growMemViews(),HEAP32)[(options+8)/4]],powerPreference:WebGPU.PowerPreference[(growMemViews(),HEAP32)[(options+12)/4]],forceFallbackAdapter:!!(growMemViews(),HEAPU32)[(options+16)/4]};var nextInChainPtr=Number((growMemViews(),HEAPU64)[options/8]);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var webxrOptions=nextInChainPtr;opts.xrCompatible=!!(growMemViews(),HEAPU32)[(webxrOptions+16)/4]}}if(!("gpu"in navigator)){var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (navigator.gpu is not available)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp);return}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,navigator.gpu.requestAdapter(opts).then(adapter=>{runtimeKeepalivePop();callUserCallback(()=>{if(adapter){WebGPU.Internals.jsObjectInsert(adapterPtr,adapter);_emwgpuOnRequestAdapterCompleted(futureId,1,BigInt(adapterPtr),0n)}else{var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (requestAdapter returned null)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)}})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestAdapterCompleted(futureId,4,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)})}))}var _emwgpuQueueOnSubmittedWorkDone=function(queuePtr,futureId){queuePtr=bigintToI53Checked(queuePtr);futureId=bigintToI53Checked(futureId);var queue=WebGPU.getJsObject(queuePtr);runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,queue.onSubmittedWorkDone().then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnWorkDoneCompleted(futureId,1)})}))};var _emwgpuWaitAny=function(futurePtr,futureCount,timeoutMSPtr){futurePtr=bigintToI53Checked(futurePtr);futureCount=bigintToI53Checked(futureCount);timeoutMSPtr=bigintToI53Checked(timeoutMSPtr);return Asyncify.handleAsync(async()=>{var promises=[];if(timeoutMSPtr){var timeoutMS=(growMemViews(),HEAP32)[timeoutMSPtr/4];promises.length=futureCount+1;promises[futureCount]=new Promise(resolve=>setTimeout(resolve,timeoutMS,0))}else{promises.length=futureCount}for(var i=0;i<futureCount;++i){var futureId=readI53FromI64(futurePtr+i*8);if(!(futureId in WebGPU.Internals.futures)){return futureId}promises[i]=WebGPU.Internals.futures[futureId]}const firstResolvedFuture=await Promise.race(promises);delete WebGPU.Internals.futures[firstResolvedFuture];return firstResolvedFuture})};_emwgpuWaitAny.isAsync=true;var ENV={};var getExecutableName=()=>thisProgram||"./this.program";var getEnvStrings=()=>{if(!getEnvStrings.strings){var lang=(typeof navigator=="object"&&navigator.language||"C").replace("-","_")+".UTF-8";var env={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:lang,_:getExecutableName()};for(var x in ENV){if(ENV[x]===undefined)delete env[x];else env[x]=ENV[x]}var strings=[];for(var x in env){strings.push(`${x}=${env[x]}`)}getEnvStrings.strings=strings}return getEnvStrings.strings};function _environ_get(__environ,environ_buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(14,0,1,__environ,environ_buf);__environ=bigintToI53Checked(__environ);environ_buf=bigintToI53Checked(environ_buf);var bufSize=0;var envp=0;for(var string of getEnvStrings()){var ptr=environ_buf+bufSize;(growMemViews(),HEAPU64)[(__environ+envp)/8]=BigInt(ptr);bufSize+=stringToUTF8(string,ptr,Infinity)+1;envp+=8}return 0}function _environ_sizes_get(penviron_count,penviron_buf_size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(15,0,1,penviron_count,penviron_buf_size);penviron_count=bigintToI53Checked(penviron_count);penviron_buf_size=bigintToI53Checked(penviron_buf_size);var strings=getEnvStrings();(growMemViews(),HEAPU64)[penviron_count/8]=BigInt(strings.length);var bufSize=0;for(var string of strings){bufSize+=lengthBytesUTF8(string)+1}(growMemViews(),HEAPU64)[penviron_buf_size/8]=BigInt(bufSize);return 0}function _fd_close(fd){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(16,0,1,fd);try{var stream=SYSCALLS.getStreamFromFD(fd);FS.close(stream);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doReadv=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.read(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len)break;if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_read(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(17,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doReadv(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _fd_seek(fd,offset,whence,newOffset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(18,0,1,fd,offset,whence,newOffset);offset=bigintToI53Checked(offset);newOffset=bigintToI53Checked(newOffset);try{if(isNaN(offset))return 61;var stream=SYSCALLS.getStreamFromFD(fd);FS.llseek(stream,offset,whence);(growMemViews(),HEAP64)[newOffset/8]=BigInt(stream.position);if(stream.getdents&&offset===0&&whence===0)stream.getdents=null;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doWritev=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.write(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len){break}if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_write(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(19,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _random_get(buffer,size){buffer=bigintToI53Checked(buffer);size=bigintToI53Checked(size);try{randomFill((growMemViews(),HEAPU8).subarray(buffer,buffer+size));return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _wgpuAdapterGetInfo(adapterPtr,info){adapterPtr=bigintToI53Checked(adapterPtr);info=bigintToI53Checked(info);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillAdapterInfoStruct(adapter.info,info);return 1}function _wgpuAdapterGetLimits(adapterPtr,limitsOutPtr){adapterPtr=bigintToI53Checked(adapterPtr);limitsOutPtr=bigintToI53Checked(limitsOutPtr);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillLimitStruct(adapter.limits,limitsOutPtr);return 1}function _wgpuAdapterHasFeature(adapterPtr,featureEnumValue){adapterPtr=bigintToI53Checked(adapterPtr);var adapter=WebGPU.getJsObject(adapterPtr);return adapter.features.has(WebGPU.FeatureName[featureEnumValue])}var _wgpuBufferGetSize=function(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);return buffer.size})();return BigInt(ret)};var _wgpuCommandEncoderBeginComputePass=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),timestampWrites:WebGPU.makePassTimestampWrites(Number((growMemViews(),HEAPU64)[(descriptor+24)/8]))}}var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateComputePassEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.beginComputePass(desc));return ptr})();return BigInt(ret)};function _wgpuCommandEncoderCopyBufferToBuffer(encoderPtr,srcPtr,srcOffset,dstPtr,dstOffset,size){encoderPtr=bigintToI53Checked(encoderPtr);srcPtr=bigintToI53Checked(srcPtr);srcOffset=bigintToI53Checked(srcOffset);dstPtr=bigintToI53Checked(dstPtr);dstOffset=bigintToI53Checked(dstOffset);size=bigintToI53Checked(size);var commandEncoder=WebGPU.getJsObject(encoderPtr);var src=WebGPU.getJsObject(srcPtr);var dst=WebGPU.getJsObject(dstPtr);commandEncoder.copyBufferToBuffer(src,srcOffset,dst,dstOffset,size)}var _wgpuCommandEncoderFinish=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateCommandBuffer(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.finish());return ptr})();return BigInt(ret)};function _wgpuComputePassEncoderDispatchWorkgroups(passPtr,x,y,z){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.dispatchWorkgroups(x,y,z)}function _wgpuComputePassEncoderEnd(passPtr){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.end()}function _wgpuComputePassEncoderSetBindGroup(passPtr,groupIndex,groupPtr,dynamicOffsetCount,dynamicOffsetsPtr){passPtr=bigintToI53Checked(passPtr);groupPtr=bigintToI53Checked(groupPtr);dynamicOffsetCount=bigintToI53Checked(dynamicOffsetCount);dynamicOffsetsPtr=bigintToI53Checked(dynamicOffsetsPtr);var pass=WebGPU.getJsObject(passPtr);var group=WebGPU.getJsObject(groupPtr);if(dynamicOffsetCount==0){pass.setBindGroup(groupIndex,group)}else{pass.setBindGroup(groupIndex,group,(growMemViews(),HEAPU32),dynamicOffsetsPtr/4,dynamicOffsetCount)}}function _wgpuComputePassEncoderSetPipeline(passPtr,pipelinePtr){passPtr=bigintToI53Checked(passPtr);pipelinePtr=bigintToI53Checked(pipelinePtr);var pass=WebGPU.getJsObject(passPtr);var pipeline=WebGPU.getJsObject(pipelinePtr);pass.setPipeline(pipeline)}var _wgpuComputePipelineGetBindGroupLayout=function(pipelinePtr,groupIndex){pipelinePtr=bigintToI53Checked(pipelinePtr);var ret=(()=>{var pipeline=WebGPU.getJsObject(pipelinePtr);var ptr=_emwgpuCreateBindGroupLayout(0n);WebGPU.Internals.jsObjectInsert(ptr,pipeline.getBindGroupLayout(groupIndex));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateBindGroup=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{function makeEntry(entryPtr){var bufferPtr=Number((growMemViews(),HEAPU64)[(entryPtr+16)/8]);var samplerPtr=Number((growMemViews(),HEAPU64)[(entryPtr+40)/8]);var textureViewPtr=Number((growMemViews(),HEAPU64)[(entryPtr+48)/8]);var externalTexturePtr=0;WebGPU.iterateExtensions(entryPtr,{327681:ptr=>{externalTexturePtr=Number((growMemViews(),HEAPU64)[(ptr+16)/8])}});var resource;if(bufferPtr){var size=readI53FromI64(entryPtr+32);if(size==-1)size=undefined;resource={buffer:WebGPU.getJsObject(bufferPtr),offset:readI53FromI64(entryPtr+24),size}}else{resource=WebGPU.getJsObject(samplerPtr||textureViewPtr||externalTexturePtr)}return{binding:(growMemViews(),HEAPU32)[(entryPtr+8)/4],resource}}function makeEntries(count,entriesPtrs){var entries=[];for(var i=0;i<count;++i){entries.push(makeEntry(entriesPtrs+56*i))}return entries}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),entries:makeEntries(Number((growMemViews(),HEAPU64)[(descriptor+32)/8]),Number((growMemViews(),HEAPU64)[(descriptor+40)/8]))};var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateBindGroup(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createBindGroup(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateCommandEncoder=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8)}}var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateCommandEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createCommandEncoder(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateComputePipeline=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc=WebGPU.makeComputePipelineDesc(descriptor);var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateComputePipeline(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createComputePipeline(desc));return ptr})();return BigInt(ret)};function _wgpuInstanceHasWGSLLanguageFeature(instance,featureEnumValue){instance=bigintToI53Checked(instance);if(!("wgslLanguageFeatures"in navigator.gpu)){return false}return navigator.gpu.wgslLanguageFeatures.has(WebGPU.WGSLLanguageFeatureName[featureEnumValue])}var _wgpuQueueSubmit=function(queuePtr,commandCount,commands){queuePtr=bigintToI53Checked(queuePtr);commandCount=bigintToI53Checked(commandCount);commands=bigintToI53Checked(commands);var queue=WebGPU.getJsObject(queuePtr);var cmds=Array.from((growMemViews(),HEAP64).subarray(commands/8,(commands+commandCount*8)/8),id=>WebGPU.getJsObject(id));queue.submit(cmds)};function _wgpuQueueWriteBuffer(queuePtr,bufferPtr,bufferOffset,data,size){queuePtr=bigintToI53Checked(queuePtr);bufferPtr=bigintToI53Checked(bufferPtr);bufferOffset=bigintToI53Checked(bufferOffset);data=bigintToI53Checked(data);size=bigintToI53Checked(size);var queue=WebGPU.getJsObject(queuePtr);var buffer=WebGPU.getJsObject(bufferPtr);var subarray=(growMemViews(),HEAPU8).subarray(data,data+size);queue.writeBuffer(buffer,bufferOffset,subarray,0,size)}var Asyncify={instrumentWasmImports(imports){var importPattern=/^(invoke_.*|__asyncjs__.*)$/;for(let[x,original]of Object.entries(imports)){if(typeof original=="function"){let isAsyncifyImport=original.isAsync||importPattern.test(x);if(isAsyncifyImport){imports[x]=original=new WebAssembly.Suspending(original)}}}},instrumentFunction(original){var wrapper=(...args)=>original(...args);return wrapper},instrumentWasmExports(exports){var exportPattern=/^(wllama_start|wllama_action|main|__main_argc_argv)$/;Asyncify.asyncExports=new Set;var ret={};for(let[x,original]of Object.entries(exports)){if(typeof original=="function"){let isAsyncifyExport=exportPattern.test(x);if(isAsyncifyExport){Asyncify.asyncExports.add(original);original=Asyncify.makeAsyncFunction(original)}var wrapper=Asyncify.instrumentFunction(original);ret[x]=wrapper}else{ret[x]=original}}return ret},asyncExports:null,isAsyncExport(func){return Asyncify.asyncExports?.has(func)},handleAsync:async startAsync=>{runtimeKeepalivePush();try{return await startAsync()}finally{runtimeKeepalivePop()}},handleSleep:startAsync=>Asyncify.handleAsync(()=>new Promise(startAsync)),makeAsyncFunction(original){return WebAssembly.promising(original)}};var getCFunc=ident=>{var func=Module["_"+ident];return func};var writeArrayToMemory=(array,buffer)=>{(growMemViews(),HEAP8).set(array,buffer)};var ccall=(ident,returnType,argTypes,args,opts)=>{var toC={pointer:p=>BigInt(p),string:str=>{var ret=0;if(str!==null&&str!==undefined&&str!==0){ret=stringToUTF8OnStack(str)}return BigInt(ret)},array:arr=>{var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return BigInt(ret)}};function convertReturnValue(ret){if(returnType==="string"){return UTF8ToString(Number(ret))}if(returnType==="pointer")return Number(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i])}else{cArgs[i]=args[i]}}}var ret=func(...cArgs);function onDone(ret){if(stack!==0)stackRestore(stack);return convertReturnValue(ret)}var asyncMode=opts?.async;if(asyncMode)return ret.then(onDone);ret=onDone(ret);return ret};var cwrap=(ident,returnType,argTypes,opts)=>{var numericArgs=!argTypes||argTypes.every(type=>type==="number"||type==="boolean");var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return(...args)=>ccall(ident,returnType,argTypes,args,opts)};var FS_createPath=(...args)=>FS.createPath(...args);var FS_unlink=(...args)=>FS.unlink(...args);var FS_createLazyFile=(...args)=>FS.createLazyFile(...args);var FS_createDevice=(...args)=>FS.createDevice(...args);PThread.init();FS.createPreloadedFile=FS_createPreloadedFile;FS.preloadFile=FS_preloadFile;FS.staticInit();{initMemory();if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(Module["preloadPlugins"])preloadPlugins=Module["preloadPlugins"];if(Module["print"])out=Module["print"];if(Module["printErr"])err=Module["printErr"];if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].shift()()}}}Module["ENV"]=ENV;Module["mmapAlloc"]=mmapAlloc;Module["wasmMemory"]=wasmMemory;Module["addRunDependency"]=addRunDependency;Module["removeRunDependency"]=removeRunDependency;Module["ccall"]=ccall;Module["cwrap"]=cwrap;Module["FS_preloadFile"]=FS_preloadFile;Module["FS_unlink"]=FS_unlink;Module["FS_createPath"]=FS_createPath;Module["FS_createDevice"]=FS_createDevice;Module["FS"]=FS;Module["FS_createDataFile"]=FS_createDataFile;Module["FS_createLazyFile"]=FS_createLazyFile;Module["MEMFS"]=MEMFS;var proxiedFunctionTable=[_proc_exit,exitOnMainThread,pthreadCreateProxied,___syscall_fcntl64,___syscall_fstat64,___syscall_getcwd,___syscall_getdents64,___syscall_ioctl,___syscall_lstat64,___syscall_newfstatat,___syscall_openat,___syscall_stat64,__mmap_js,__munmap_js,_environ_get,_environ_sizes_get,_fd_close,_fd_read,_fd_seek,_fd_write];function __asyncjs__js_file_read(path_ptr,offset,req_size,out_ptr){return Asyncify.handleAsync(async()=>await _wllama_js_file_read(UTF8ToString(Number(path_ptr)),Number(offset),Number(req_size),Number(out_ptr)))}__asyncjs__js_file_read.sig="jjjjj";var _wllama_malloc,_wllama_start,_wllama_action,_wllama_exit,_wllama_debug,_main,_malloc,_free,_emwgpuCreateBindGroup,_emwgpuCreateBindGroupLayout,_emwgpuCreateCommandBuffer,_emwgpuCreateCommandEncoder,_emwgpuCreateComputePassEncoder,_emwgpuCreateComputePipeline,_emwgpuCreateExternalTexture,_emwgpuCreatePipelineLayout,_emwgpuCreateQuerySet,_emwgpuCreateRenderBundle,_emwgpuCreateRenderBundleEncoder,_emwgpuCreateRenderPassEncoder,_emwgpuCreateRenderPipeline,_emwgpuCreateSampler,_emwgpuCreateSurface,_emwgpuCreateTexture,_emwgpuCreateTextureView,_emwgpuCreateAdapter,_emwgpuCreateBuffer,_emwgpuCreateDevice,_emwgpuCreateQueue,_emwgpuCreateShaderModule,_emwgpuOnDeviceLostCompleted,_emwgpuOnMapAsyncCompleted,_emwgpuOnRequestAdapterCompleted,_emwgpuOnRequestDeviceCompleted,_emwgpuOnWorkDoneCompleted,_emwgpuOnUncapturedError,__emscripten_tls_init,_pthread_self,_emscripten_builtin_memalign,__emscripten_thread_init,__emscripten_thread_crashed,__emscripten_run_js_on_main_thread,__emscripten_thread_free_data,__emscripten_thread_exit,__emscripten_check_mailbox,_memalign,___trap,_emscripten_stack_set_limits,__emscripten_stack_restore,__emscripten_stack_alloc,_emscripten_stack_get_current,__indirect_function_table,wasmTable;function assignWasmExports(wasmExports){_wllama_malloc=Module["_wllama_malloc"]=wasmExports["wllama_malloc"];_wllama_start=Module["_wllama_start"]=wasmExports["wllama_start"];_wllama_action=Module["_wllama_action"]=wasmExports["wllama_action"];_wllama_exit=Module["_wllama_exit"]=wasmExports["wllama_exit"];_wllama_debug=Module["_wllama_debug"]=wasmExports["wllama_debug"];_main=Module["_main"]=wasmExports["main"];_malloc=wasmExports["malloc"];_free=wasmExports["free"];_emwgpuCreateBindGroup=wasmExports["emwgpuCreateBindGroup"];_emwgpuCreateBindGroupLayout=wasmExports["emwgpuCreateBindGroupLayout"];_emwgpuCreateCommandBuffer=wasmExports["emwgpuCreateCommandBuffer"];_emwgpuCreateCommandEncoder=wasmExports["emwgpuCreateCommandEncoder"];_emwgpuCreateComputePassEncoder=wasmExports["emwgpuCreateComputePassEncoder"];_emwgpuCreateComputePipeline=wasmExports["emwgpuCreateComputePipeline"];_emwgpuCreateExternalTexture=wasmExports["emwgpuCreateExternalTexture"];_emwgpuCreatePipelineLayout=wasmExports["emwgpuCreatePipelineLayout"];_emwgpuCreateQuerySet=wasmExports["emwgpuCreateQuerySet"];_emwgpuCreateRenderBundle=wasmExports["emwgpuCreateRenderBundle"];_emwgpuCreateRenderBundleEncoder=wasmExports["emwgpuCreateRenderBundleEncoder"];_emwgpuCreateRenderPassEncoder=wasmExports["emwgpuCreateRenderPassEncoder"];_emwgpuCreateRenderPipeline=wasmExports["emwgpuCreateRenderPipeline"];_emwgpuCreateSampler=wasmExports["emwgpuCreateSampler"];_emwgpuCreateSurface=wasmExports["emwgpuCreateSurface"];_emwgpuCreateTexture=wasmExports["emwgpuCreateTexture"];_emwgpuCreateTextureView=wasmExports["emwgpuCreateTextureView"];_emwgpuCreateAdapter=wasmExports["emwgpuCreateAdapter"];_emwgpuCreateBuffer=wasmExports["emwgpuCreateBuffer"];_emwgpuCreateDevice=wasmExports["emwgpuCreateDevice"];_emwgpuCreateQueue=wasmExports["emwgpuCreateQueue"];_emwgpuCreateShaderModule=wasmExports["emwgpuCreateShaderModule"];_emwgpuOnDeviceLostCompleted=wasmExports["emwgpuOnDeviceLostCompleted"];_emwgpuOnMapAsyncCompleted=wasmExports["emwgpuOnMapAsyncCompleted"];_emwgpuOnRequestAdapterCompleted=wasmExports["emwgpuOnRequestAdapterCompleted"];_emwgpuOnRequestDeviceCompleted=wasmExports["emwgpuOnRequestDeviceCompleted"];_emwgpuOnWorkDoneCompleted=wasmExports["emwgpuOnWorkDoneCompleted"];_emwgpuOnUncapturedError=wasmExports["emwgpuOnUncapturedError"];__emscripten_tls_init=wasmExports["_emscripten_tls_init"];_pthread_self=wasmExports["pthread_self"];_emscripten_builtin_memalign=wasmExports["emscripten_builtin_memalign"];__emscripten_thread_init=wasmExports["_emscripten_thread_init"];__emscripten_thread_crashed=wasmExports["_emscripten_thread_crashed"];__emscripten_run_js_on_main_thread=wasmExports["_emscripten_run_js_on_main_thread"];__emscripten_thread_free_data=wasmExports["_emscripten_thread_free_data"];__emscripten_thread_exit=wasmExports["_emscripten_thread_exit"];__emscripten_check_mailbox=wasmExports["_emscripten_check_mailbox"];_memalign=wasmExports["memalign"];___trap=wasmExports["__trap"];_emscripten_stack_set_limits=wasmExports["emscripten_stack_set_limits"];__emscripten_stack_restore=wasmExports["_emscripten_stack_restore"];__emscripten_stack_alloc=wasmExports["_emscripten_stack_alloc"];_emscripten_stack_get_current=wasmExports["emscripten_stack_get_current"];__indirect_function_table=wasmTable=wasmExports["__indirect_function_table"]}var wasmImports;function assignWasmImports(){wasmImports={__asyncjs__js_file_read,__pthread_create_js:___pthread_create_js,__syscall_fcntl64:___syscall_fcntl64,__syscall_getcwd:___syscall_getcwd,__syscall_getdents64:___syscall_getdents64,__syscall_ioctl:___syscall_ioctl,__syscall_openat:___syscall_openat,__syscall_stat64:___syscall_stat64,_abort_js:__abort_js,_emscripten_init_main_thread_js:__emscripten_init_main_thread_js,_emscripten_notify_mailbox_postmessage:__emscripten_notify_mailbox_postmessage,_emscripten_receive_on_main_thread_js:__emscripten_receive_on_main_thread_js,_emscripten_thread_cleanup:__emscripten_thread_cleanup,_emscripten_thread_mailbox_await:__emscripten_thread_mailbox_await,_emscripten_thread_set_strongref:__emscripten_thread_set_strongref,_localtime_js:__localtime_js,_mmap_js:__mmap_js,_munmap_js:__munmap_js,_tzset_js:__tzset_js,clock_time_get:_clock_time_get,emscripten_check_blocking_allowed:_emscripten_check_blocking_allowed,emscripten_date_now:_emscripten_date_now,emscripten_exit_with_live_runtime:_emscripten_exit_with_live_runtime,emscripten_get_callstack:_emscripten_get_callstack,emscripten_get_heap_max:_emscripten_get_heap_max,emscripten_get_now:_emscripten_get_now,emscripten_has_asyncify:_emscripten_has_asyncify,emscripten_num_logical_cores:_emscripten_num_logical_cores,emscripten_resize_heap:_emscripten_resize_heap,emwgpuAdapterRequestDevice:_emwgpuAdapterRequestDevice,emwgpuBufferDestroy:_emwgpuBufferDestroy,emwgpuBufferGetConstMappedRange:_emwgpuBufferGetConstMappedRange,emwgpuBufferMapAsync:_emwgpuBufferMapAsync,emwgpuBufferUnmap:_emwgpuBufferUnmap,emwgpuDelete:_emwgpuDelete,emwgpuDeviceCreateBuffer:_emwgpuDeviceCreateBuffer,emwgpuDeviceCreateShaderModule:_emwgpuDeviceCreateShaderModule,emwgpuDeviceDestroy:_emwgpuDeviceDestroy,emwgpuInstanceRequestAdapter:_emwgpuInstanceRequestAdapter,emwgpuQueueOnSubmittedWorkDone:_emwgpuQueueOnSubmittedWorkDone,emwgpuWaitAny:_emwgpuWaitAny,environ_get:_environ_get,environ_sizes_get:_environ_sizes_get,exit:_exit,fd_close:_fd_close,fd_read:_fd_read,fd_seek:_fd_seek,fd_write:_fd_write,memory:wasmMemory,random_get:_random_get,wgpuAdapterGetInfo:_wgpuAdapterGetInfo,wgpuAdapterGetLimits:_wgpuAdapterGetLimits,wgpuAdapterHasFeature:_wgpuAdapterHasFeature,wgpuBufferGetSize:_wgpuBufferGetSize,wgpuCommandEncoderBeginComputePass:_wgpuCommandEncoderBeginComputePass,wgpuCommandEncoderCopyBufferToBuffer:_wgpuCommandEncoderCopyBufferToBuffer,wgpuCommandEncoderFinish:_wgpuCommandEncoderFinish,wgpuComputePassEncoderDispatchWorkgroups:_wgpuComputePassEncoderDispatchWorkgroups,wgpuComputePassEncoderEnd:_wgpuComputePassEncoderEnd,wgpuComputePassEncoderSetBindGroup:_wgpuComputePassEncoderSetBindGroup,wgpuComputePassEncoderSetPipeline:_wgpuComputePassEncoderSetPipeline,wgpuComputePipelineGetBindGroupLayout:_wgpuComputePipelineGetBindGroupLayout,wgpuDeviceCreateBindGroup:_wgpuDeviceCreateBindGroup,wgpuDeviceCreateCommandEncoder:_wgpuDeviceCreateCommandEncoder,wgpuDeviceCreateComputePipeline:_wgpuDeviceCreateComputePipeline,wgpuInstanceHasWGSLLanguageFeature:_wgpuInstanceHasWGSLLanguageFeature,wgpuQueueSubmit:_wgpuQueueSubmit,wgpuQueueWriteBuffer:_wgpuQueueWriteBuffer}}function applySignatureConversions(wasmExports){wasmExports=Object.assign({},wasmExports);var makeWrapper___PP=f=>(a0,a1,a2)=>f(a0,BigInt(a1?a1:0),BigInt(a2?a2:0));var makeWrapper_pp=f=>a0=>Number(f(BigInt(a0)));var makeWrapper__p=f=>a0=>f(BigInt(a0));var makeWrapper_p=f=>()=>Number(f());var makeWrapper_ppp=f=>(a0,a1)=>Number(f(BigInt(a0),BigInt(a1)));var makeWrapper__p_____=f=>(a0,a1,a2,a3,a4,a5)=>f(BigInt(a0),a1,a2,a3,a4,a5);var makeWrapper___p_p_=f=>(a0,a1,a2,a3,a4)=>f(a0,BigInt(a1),a2,BigInt(a3),a4);var makeWrapper__pp=f=>(a0,a1)=>f(BigInt(a0),BigInt(a1));wasmExports["main"]=makeWrapper___PP(wasmExports["main"]);wasmExports["malloc"]=makeWrapper_pp(wasmExports["malloc"]);wasmExports["free"]=makeWrapper__p(wasmExports["free"]);wasmExports["pthread_self"]=makeWrapper_p(wasmExports["pthread_self"]);wasmExports["emscripten_builtin_memalign"]=makeWrapper_ppp(wasmExports["emscripten_builtin_memalign"]);wasmExports["_emscripten_thread_init"]=makeWrapper__p_____(wasmExports["_emscripten_thread_init"]);wasmExports["_emscripten_run_js_on_main_thread"]=makeWrapper___p_p_(wasmExports["_emscripten_run_js_on_main_thread"]);wasmExports["_emscripten_thread_free_data"]=makeWrapper__p(wasmExports["_emscripten_thread_free_data"]);wasmExports["_emscripten_thread_exit"]=makeWrapper__p(wasmExports["_emscripten_thread_exit"]);wasmExports["memalign"]=makeWrapper_ppp(wasmExports["memalign"]);wasmExports["emscripten_stack_set_limits"]=makeWrapper__pp(wasmExports["emscripten_stack_set_limits"]);wasmExports["_emscripten_stack_restore"]=makeWrapper__p(wasmExports["_emscripten_stack_restore"]);wasmExports["_emscripten_stack_alloc"]=makeWrapper_pp(wasmExports["_emscripten_stack_alloc"]);wasmExports["emscripten_stack_get_current"]=makeWrapper_p(wasmExports["emscripten_stack_get_current"]);return wasmExports}async function callMain(){var entryFunction=_main;var argc=0;var argv=0;try{var ret=entryFunction(argc,BigInt(argv));ret=await ret;exitJS(ret,true);return ret}catch(e){return handleException(e)}}function run(){if(runDependencies>0){dependenciesFulfilled=run;return}if(ENVIRONMENT_IS_PTHREAD){initRuntime();return}preRun();if(runDependencies>0){dependenciesFulfilled=run;return}async function doRun(){Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();Module["onRuntimeInitialized"]?.();var noInitialRun=Module["noInitialRun"]||false;if(!noInitialRun)await callMain();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun()},1)}else{doRun()}}var wasmExports;if(!ENVIRONMENT_IS_PTHREAD){createWasm();run()}\n';

// src/worker.ts
var FILE_READ_REQ_EVENT = "fs.read_req";
var JSPI_STUB = `
if (!WebAssembly.Suspending) {
  // JSPI not available - stubs that keep the import/export tables valid.
  // Suspending wraps imports: identity is fine since async imports won't be called.
  WebAssembly.Suspending = function (fn) {
    // console.log(fn.toString());
    return fn;
  };
  // promising wraps exports: must return a Promise so ccall's ret.then() works.
  WebAssembly.promising = function (fn) {
    return function (...args) {
      try {
        return Promise.resolve(fn(...args));
      } catch (e) {
        return Promise.reject(e);
      }
    };
  };
}
`;
var ProxyToWorker = class {
  // filename -> Blob for async reads
  constructor(resources, nbThread, suppressNativeLog, logger) {
    __publicField(this, "resources");
    __publicField(this, "logger");
    __publicField(this, "suppressNativeLog");
    __publicField(this, "taskQueue", []);
    __publicField(this, "taskId", 1);
    __publicField(this, "resultQueue", []);
    __publicField(this, "busy", false);
    // is the work loop is running?
    __publicField(this, "worker");
    __publicField(this, "multiThread");
    __publicField(this, "nbThread");
    __publicField(this, "useAsyncFile");
    __publicField(this, "fileBlobs", /* @__PURE__ */ new Map());
    this.resources = resources;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 0;
    this.logger = logger;
    this.suppressNativeLog = suppressNativeLog;
    this.useAsyncFile = canUseAsyncFileRead(resources.compat);
  }
  getModuleCode() {
    return __async(this, null, function* () {
      if (!this.resources.jsPath) {
        if (this.resources.compat) {
          throw new Error(
            "compat mode is enabled but no jsPath was provided. Pass a worker JS via setCompat() or install @wllama/wllama-compat."
          );
        }
        return WLLAMA_EMSCRIPTEN_CODE;
      } else if (this.resources.jsPath.code) {
        return this.resources.jsPath.code;
      } else if (isString(this.resources.jsPath)) {
        const response = yield fetch(this.resources.jsPath);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch worker code from ${this.resources.jsPath}`
          );
        }
        return yield response.text();
      } else {
        throw new Error("No JS code provided for worker");
      }
    });
  }
  moduleInit(ggufFiles) {
    return __async(this, null, function* () {
      let moduleCode = JSPI_STUB + (yield this.getModuleCode());
      if (this.resources.noWebGPU) {
        moduleCode = 'try{Object.defineProperty(WorkerNavigator.prototype,"gpu",{get:()=>({requestAdapter:async()=>null})});}catch(e){}' + moduleCode;
      }
      let mainModuleCode = moduleCode.replace("var Module", "var ___Module");
      const runOptions = {
        pathConfig: {
          "wllama.wasm": this.resources.wasmPath
        },
        nbThread: this.nbThread,
        compat: this.resources.compat
      };
      const completeCode = [
        `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
        `function wModuleInit() { ${mainModuleCode}; return Module; }`,
        LLAMA_CPP_WORKER_CODE
      ].join(";\n\n");
      this.worker = createWorker(completeCode);
      this.worker.onmessage = this.onRecvMsg.bind(this);
      this.worker.onerror = this.logger.error;
      const res = yield this.pushTask({
        verb: "module.init",
        args: [
          new Blob([moduleCode], { type: "text/javascript" }),
          this.useAsyncFile
        ],
        callbackId: this.taskId++
      });
      const nativeFiles = [];
      for (const file of ggufFiles) {
        const needAllocBuffer = !this.useAsyncFile;
        const id = yield this.fileAlloc(
          file.name,
          file.blob.size,
          needAllocBuffer
        );
        nativeFiles.push(__spreadValues({ id }, file));
        if (this.useAsyncFile) {
          this.fileBlobs.set(file.name, file.blob);
        }
      }
      if (!this.useAsyncFile) {
        yield Promise.all(
          nativeFiles.map((file) => {
            return this.fileWrite(file.id, file.blob);
          })
        );
      }
      return res;
    });
  }
  wllamaStart() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.start",
        args: [],
        callbackId: this.taskId++
      });
      const parsedResult = this.parseResult(result);
      return parsedResult;
    });
  }
  wllamaAction(name, body) {
    return __async(this, null, function* () {
      const encodedMsg = glueSerialize(body);
      const result = yield this.pushTask({
        verb: "wllama.action",
        args: [name, encodedMsg],
        callbackId: this.taskId++
      });
      const parsedResult = glueDeserialize(result);
      if (parsedResult._name === "erro_evt") {
        throw new WllamaRuntimeError(parsedResult.message, "");
      }
      return parsedResult;
    });
  }
  /** Take a slot save file out of the worker's in-memory FS (the file is removed) */
  slotFileRead(fileName) {
    return __async(this, null, function* () {
      return yield this.pushTask({
        verb: "fs.slot_read",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  /** Put a slot save file into the worker's in-memory FS */
  slotFileWrite(fileName, data) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_write",
        args: [fileName, data],
        callbackId: this.taskId++
      });
    });
  }
  slotFileRemove(fileName) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_remove",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  wllamaExit() {
    return __async(this, null, function* () {
      if (this.worker) {
        this.worker.terminate();
      }
    });
  }
  wllamaDebug() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.debug",
        args: [],
        callbackId: this.taskId++
      });
      return JSON.parse(result);
    });
  }
  ///////////////////////////////////////
  /**
   * Allocate a new file in heapfs
   * @returns fileId, to be used by fileWrite()
   */
  fileAlloc(fileName, size, allocBuffer) {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "fs.alloc",
        args: [fileName, size, allocBuffer],
        callbackId: this.taskId++
      });
      return result.fileId;
    });
  }
  /**
   * Write a Blob to heapfs
   */
  fileWrite(fileId, blob) {
    return __async(this, null, function* () {
      const reader = blob.stream().getReader();
      let offset = 0;
      while (true) {
        const { done, value } = yield reader.read();
        if (done) break;
        const size = value.byteLength;
        yield this.pushTask(
          {
            verb: "fs.write",
            args: [fileId, value, offset],
            callbackId: this.taskId++
          },
          // @ts-ignore Type 'ArrayBufferLike' is not assignable to type 'ArrayBuffer'
          [value.buffer]
        );
        offset += size;
      }
    });
  }
  fileReadResponse(name, offset, size) {
    return __async(this, null, function* () {
      var _a;
      try {
        const blob = this.fileBlobs.get(name);
        if (!blob) {
          throw new Error(`blob not found for name="${name}"`);
        }
        const chunk = blob.slice(offset, offset + size);
        const buffer = yield chunk.arrayBuffer();
        this.worker.postMessage(
          { verb: "fs.read_res", args: [buffer] },
          { transfer: [buffer] }
        );
      } catch (err) {
        this.logger.error("fileReadResponse failed, terminating worker:", err);
        (_a = this.worker) == null ? void 0 : _a.terminate();
        this.worker = void 0;
        this.abort(`File read failed: ${err}`, err.stack || "");
      }
    });
  }
  /**
   * Parse JSON result returned by cpp code.
   * Throw new Error if "__exception" is present in the response
   *
   * TODO: get rid of this function once everything is migrated to Glue
   */
  parseResult(result) {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult["error"]) {
      throw new WllamaRuntimeError("Unknown error, please see console.log", "");
    }
    return parsedResult;
  }
  /**
   * Push a new task to taskQueue
   */
  pushTask(param, buffers) {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ resolve, reject, param, buffers });
      this.runTaskLoop();
    });
  }
  /**
   * Main loop for processing tasks
   */
  runTaskLoop() {
    return __async(this, null, function* () {
      var _a;
      if (this.busy) {
        return;
      }
      this.busy = true;
      while (true) {
        const task = this.taskQueue.shift();
        if (!task) break;
        this.resultQueue.push(task);
        this.worker.postMessage(
          task.param,
          isSafariMobile() ? void 0 : {
            transfer: (_a = task.buffers) != null ? _a : []
          }
        );
      }
      this.busy = false;
    });
  }
  /**
   * Handle messages from worker
   */
  onRecvMsg(e) {
    if (!e.data) return;
    const { verb, args } = e.data;
    const isCompatBuild = this.resources.compat;
    if (verb && verb.startsWith("console.")) {
      if (this.suppressNativeLog) {
        return;
      }
      if (verb.endsWith("debug")) this.logger.debug(...args);
      if (verb.endsWith("log")) this.logger.log(...args);
      if (verb.endsWith("warn")) this.logger.warn(...args);
      if (verb.endsWith("error")) this.logger.error(...args);
      return;
    } else if (verb === "signal.abort") {
      const [signalType, message, rawStack, originalErr] = args;
      if (originalErr) {
        this.logger.error(originalErr);
      }
      (() => __async(this, null, function* () {
        let stack = "";
        let newMsg = message.replace(
          "Build with -sASSERTIONS for more info.",
          ""
        );
        if (signalType === "abort") {
          newMsg = `(ABORT) ${newMsg}`;
          stack = rawStack.replace(/\|/g, "\n");
        } else if (signalType === "exception") {
          stack = rawStack;
        }
        const decoded = yield Debug.decodeStackTrace(stack, isCompatBuild);
        this.logger.error(`Stack trace (${signalType}):
` + decoded);
        this.abort(newMsg, decoded);
      }))();
      return;
    }
    if (verb === FILE_READ_REQ_EVENT) {
      const [name, offset, size] = args;
      this.fileReadResponse(name, offset, size).catch(() => {
      });
      return;
    }
    const { callbackId, result, err } = e.data;
    if (callbackId) {
      const idx = this.resultQueue.findIndex(
        (t) => t.param.callbackId === callbackId
      );
      if (idx !== -1) {
        const waitingTask = this.resultQueue.splice(idx, 1)[0];
        if (err) waitingTask.reject(err);
        else waitingTask.resolve(result);
      } else {
        this.logger.error(
          `Cannot find waiting task with callbackId = ${callbackId}`
        );
      }
    }
  }
  abort(text, stack) {
    const error = new WllamaRuntimeError(
      text.length == 0 ? "(unknown error)" : text,
      stack
    );
    while (this.resultQueue.length > 0) {
      const waitingTask = this.resultQueue.pop();
      if (!waitingTask) break;
      waitingTask.reject(error);
    }
    while (this.taskQueue.length > 0) {
      const pendingTask = this.taskQueue.pop();
      if (!pendingTask) break;
      pendingTask.reject(error);
    }
  }
};

// src/huggingface.ts
var HF_BASE = "https://huggingface.co";
var DEFAULT_QUANTS = ["Q4_K_M", "Q8_0"];
function fetchRepoFiles(repo, token) {
  return __async(this, null, function* () {
    var _a;
    const url = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
    const headers = { Accept: "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const res = yield fetch(url, { headers });
    if (!res.ok) {
      let msg = res.statusText;
      try {
        msg = (_a = (yield res.json()).error) != null ? _a : msg;
      } catch (e) {
      }
      throw new Error(`HF API error (${res.status}): ${msg}`);
    }
    return res.json();
  });
}
function firstShardPath(files, path) {
  const m = path.match(/^(.+)-(\d{5})-of-(\d{5})\.gguf$/i);
  if (!m) return path;
  const first = `${m[1]}-00001-of-${m[3]}.gguf`;
  return files.some((f) => f.path === first) ? first : path;
}
function selectFile(files, quant, mmprojOnly) {
  const candidates = files.filter((f) => {
    if (f.type !== "file" || !f.path.toLowerCase().endsWith(".gguf"))
      return false;
    const ismmproj = f.path.toLowerCase().includes("mmproj");
    return mmprojOnly ? ismmproj : !ismmproj;
  });
  if (candidates.length === 0) return null;
  if (quant) {
    const upper = quant.toUpperCase();
    const match = candidates.find((f) => f.path.toUpperCase().includes(upper));
    if (match) return firstShardPath(candidates, match.path);
    return null;
  }
  for (const q of DEFAULT_QUANTS) {
    const match = candidates.find((f) => f.path.toUpperCase().includes(q));
    if (match) return firstShardPath(candidates, match.path);
  }
  return firstShardPath(candidates, candidates[0].path);
}
function getHFModelSource(config) {
  return __async(this, null, function* () {
    const { repo, file, quant, mmprojFile, mmprojQuant, hfToken } = config;
    const files = yield fetchRepoFiles(repo, hfToken);
    const modelPath = file != null ? file : selectFile(files, quant, false);
    if (!modelPath) {
      throw new Error(`No GGUF file found in repo "${repo}"`);
    }
    const source = {
      url: `${HF_BASE}/${repo}/resolve/main/${modelPath}`
    };
    if (mmprojFile || mmprojQuant !== void 0) {
      const mmpath = mmprojFile != null ? mmprojFile : selectFile(files, mmprojQuant, true);
      if (mmpath) {
        source.mmprojUrl = `${HF_BASE}/${repo}/resolve/main/${mmpath}`;
      }
    }
    if (hfToken) {
      const params = new URLSearchParams({ token: hfToken });
      source.url += `?${params}`;
      if (source.mmprojUrl) {
        source.mmprojUrl += `?${params}`;
      }
    }
    return source;
  });
}
function getHFFileSHA256(url, headers) {
  return __async(this, null, function* () {
    if (!url.includes("/resolve/")) return void 0;
    const rawUrl = url.replace("/resolve/", "/raw/");
    try {
      const text = yield fetch(rawUrl, { headers }).then((r) => r.text());
      const match = text.match(/^oid sha256:([0-9a-f]{64})$/m);
      return match ? match[1] : void 0;
    } catch (e) {
      return void 0;
    }
  });
}

// src/storage/opfs.ts
var OPFSBackend = class {
  isSupported() {
    var _a;
    return typeof navigator !== "undefined" && "storage" in navigator && !!((_a = navigator.storage) == null ? void 0 : _a.getDirectory);
  }
  read(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        return yield fileHandle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  write(key, stream) {
    return __async(this, null, function* () {
      const writable = yield openWritable(key);
      yield writable.truncate(0);
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        const file = yield fileHandle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      const cacheDir = yield getCacheDir();
      const result = [];
      try {
        for (var iter = __forAwait(cacheDir.entries()), more, temp, error; more = !(temp = yield iter.next()).done; more = false) {
          const [name, handle] = temp.value;
          if (handle.kind === "file") {
            const file = yield handle.getFile();
            result.push({ key: name, size: file.size });
          }
        }
      } catch (temp) {
        error = [temp];
      } finally {
        try {
          more && (temp = iter.return) && (yield temp.call(iter));
        } finally {
          if (error)
            throw error[0];
        }
      }
      return result;
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        yield cacheDir.removeEntry(key);
      } catch (e) {
        if ((e == null ? void 0 : e.name) !== "NotFoundError") throw e;
      }
    });
  }
};
function getCacheDir() {
  return __async(this, null, function* () {
    const opfsRoot = yield navigator.storage.getDirectory();
    return opfsRoot.getDirectoryHandle("cache", { create: true });
  });
}
function openWritable(fileName) {
  return __async(this, null, function* () {
    const worker = createWorker(OPFS_UTILS_WORKER_CODE);
    let pResolve;
    let pReject;
    worker.onmessage = (e) => {
      if (e.data.ok) pResolve(null);
      else if (e.data.err) pReject(e.data.err);
    };
    worker.onerror = (e) => {
      var _a;
      return pReject == null ? void 0 : pReject((_a = e.message) != null ? _a : e);
    };
    const workerExec = (data) => new Promise((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      worker.postMessage(
        data,
        isSafariMobile() ? void 0 : { transfer: "buf" in data && data.buf ? [data.buf.buffer] : [] }
      );
    });
    yield workerExec({ action: "open", filename: fileName });
    return {
      truncate: () => __async(this, null, function* () {
      }),
      write: (value) => workerExec({ action: "write", buf: value }),
      close: () => __async(this, null, function* () {
        yield workerExec({ action: "close" });
        worker.terminate();
      })
    };
  });
}

// src/storage/cos.ts
function makeHash(key) {
  return { algorithm: "SHA-256", value: key };
}
var COSInternalBackend = class {
  isSupported() {
    return typeof navigator !== "undefined" && "crossOriginStorage" in navigator;
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  read(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        return handle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  write(key, stream) {
    return __async(this, null, function* () {
      const handle = yield navigator.crossOriginStorage.requestFileHandle(
        makeHash(key),
        { create: true }
      );
      const writable = yield handle.createWritable();
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        const file = yield handle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
  delete(_key) {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
};
var COSBackend = class {
  constructor() {
    __publicField(this, "cos", new COSInternalBackend());
    __publicField(this, "priv", new OPFSBackend());
  }
  isSupported() {
    return this.priv.isSupported();
  }
  read(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const blob = yield this.cos.read(hint.sha256);
        if (blob) return blob;
      }
      return this.priv.read(key);
    });
  }
  write(key, stream, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        yield this.cos.write(hint.sha256, stream);
      } else {
        yield this.priv.write(key, stream);
      }
    });
  }
  getSize(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const size = yield this.cos.getSize(hint.sha256);
        if (size !== -1) return size;
      }
      return this.priv.getSize(key);
    });
  }
  list() {
    return __async(this, null, function* () {
      return this.priv.list();
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      return this.priv.delete(key);
    });
  }
};

// src/cache-manager.ts
var PREFIX_METADATA = "__metadata__";
var POLYFILL_ETAG = "polyfill_for_older_version";
function hintFromMetadata(metadata) {
  if (!metadata) return void 0;
  if (metadata.sha256) return { sha256: metadata.sha256 };
  return void 0;
}
var CacheManager = class {
  /**
   * @param backends Array of storage backends to use, in order of preference ; if first is available, use it, otherwise try the next one.
   */
  constructor(backends = [new COSBackend()]) {
    __publicField(this, "sb");
    for (const backend of backends) {
      if (backend.isSupported()) {
        this.sb = backend;
        return;
      }
    }
    throw new Error("No supported storage backend found");
  }
  /**
   * Convert a given URL into a storage key.
   *
   * Format: `${hashSHA1(fullURL)}_${fileName}`
   */
  getNameFromURL(url) {
    return __async(this, null, function* () {
      return urlToFileName(url, "");
    });
  }
  /**
   * @deprecated Use `download()` instead
   *
   * Write a new file to cache. This will overwrite existing file.
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   */
  write(name, stream, metadata) {
    return __async(this, null, function* () {
      yield this.sb.write(name, stream);
      yield this.writeMetadata(name, metadata);
    });
  }
  download(_0) {
    return __async(this, arguments, function* (url, options = {}) {
      var _a, _b, _c, _d;
      const fileKey = yield urlToFileName(url, "");
      const sha256 = yield getHFFileSHA256(url, (_a = options.headers) != null ? _a : {});
      const hint = sha256 ? { sha256 } : void 0;
      const cachedSize = yield this.sb.getSize(fileKey, hint);
      if (cachedSize !== -1) {
        const metadata2 = yield this.readMetadata(fileKey);
        if ((metadata2 == null ? void 0 : metadata2.originalURL) === url && metadata2.originalSize === cachedSize) {
          return;
        }
        const head = yield fetch(url, __spreadValues({
          method: "HEAD"
        }, options.headers ? { headers: options.headers } : {}));
        const originalSize = parseInt(
          (_b = head.headers.get("content-length")) != null ? _b : "0",
          10
        );
        const etag2 = (head.headers.get("etag") || "").replace(
          /[^A-Za-z0-9]/g,
          ""
        );
        if (originalSize > 0 && originalSize === cachedSize) {
          yield this.writeMetadata(fileKey, __spreadValues({
            originalURL: url,
            originalSize,
            etag: etag2,
            sha256
          }, (_c = options.metadataAdditional) != null ? _c : {}));
          return;
        }
        yield this.sb.delete(fileKey);
        yield this.sb.delete(`${PREFIX_METADATA}${fileKey}`);
      }
      const response = yield fetch(url, __spreadValues(__spreadValues({}, options.headers ? { headers: options.headers } : {}), options.signal ? { signal: options.signal } : {}));
      if (!response.ok || !response.body) {
        throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
      }
      const contentLength = response.headers.get("content-length");
      const etag = (response.headers.get("etag") || "").replace(
        /[^A-Za-z0-9]/g,
        ""
      );
      const total = parseInt(contentLength != null ? contentLength : "0", 10);
      const progressCallback = options.progressCallback;
      let loaded = 0;
      let lastProgressAt = 0;
      const progressStream = new TransformStream({
        transform(chunk, controller) {
          loaded += chunk.byteLength;
          if (progressCallback) {
            const now = Date.now();
            if (now - lastProgressAt > 100) {
              lastProgressAt = now;
              progressCallback({ loaded, total });
            }
          }
          controller.enqueue(chunk);
        },
        flush() {
          progressCallback == null ? void 0 : progressCallback({ loaded, total: total || loaded });
        }
      });
      const metadata = __spreadValues({
        originalURL: url,
        originalSize: total,
        etag
      }, (_d = options.metadataAdditional) != null ? _d : {});
      if (sha256) {
        metadata.sha256 = sha256;
      }
      yield this.sb.write(
        fileKey,
        response.body.pipeThrough(progressStream),
        hint
      );
      yield this.writeMetadata(fileKey, metadata);
    });
  }
  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  open(nameOrURL) {
    return __async(this, null, function* () {
      const hint1 = hintFromMetadata(yield this.getMetadata(nameOrURL));
      const direct = yield this.sb.read(nameOrURL, hint1);
      if (direct) return direct;
      const key = yield urlToFileName(nameOrURL, "");
      const hint2 = hintFromMetadata(yield this.getMetadata(key));
      return this.sb.read(key, hint2);
    });
  }
  /**
   * Get the size of a file in stored cache
   *
   * NOTE: in case the download is stopped mid-way (i.e. user close browser tab), the file maybe corrupted, size maybe different from `metadata.originalSize`
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   * @returns number of bytes, or -1 if file does not exist
   */
  getSize(name) {
    return __async(this, null, function* () {
      const hint = hintFromMetadata(yield this.getMetadata(name));
      return this.sb.getSize(name, hint);
    });
  }
  /**
   * Get metadata of a cached file
   */
  getMetadata(name) {
    return __async(this, null, function* () {
      const metadata = yield this.readMetadata(name);
      if (metadata) return metadata;
      const cachedSize = yield this.sb.getSize(name);
      return cachedSize > 0 ? (
        // files created by older version of wllama don't have metadata; polyfill it
        {
          etag: POLYFILL_ETAG,
          originalSize: cachedSize,
          originalURL: ""
        }
      ) : (
        // cached file not found
        null
      );
    });
  }
  /**
   * Same as `getMetadata()`, but without polyfill. Returns null if the file has no metadata.
   */
  readMetadata(name) {
    return __async(this, null, function* () {
      const blob = yield this.sb.read(`${PREFIX_METADATA}${name}`);
      if (!blob) return null;
      try {
        return yield new Response(blob).json();
      } catch (e) {
        return null;
      }
    });
  }
  /**
   * List all files currently in cache
   */
  list() {
    return __async(this, null, function* () {
      const all = yield this.sb.list();
      const metadataMap = {};
      for (const { key } of all) {
        if (key.startsWith(PREFIX_METADATA)) {
          const blob = yield this.sb.read(key);
          if (blob) {
            const meta = yield new Response(blob).json().catch(() => null);
            metadataMap[key.slice(PREFIX_METADATA.length)] = meta;
          }
        }
      }
      const result = [];
      for (const { key, size } of all) {
        if (!key.startsWith(PREFIX_METADATA)) {
          result.push({
            name: key,
            size,
            metadata: metadataMap[key] || {
              originalSize: size,
              originalURL: "",
              etag: ""
            }
          });
        }
      }
      return result;
    });
  }
  /**
   * Clear all files currently in cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.deleteMany(() => true);
    });
  }
  /**
   * Delete a single file in cache
   *
   * @param nameOrURL Can be either an URL or a name returned by `getNameFromURL()` or `list()`
   */
  delete(nameOrURL) {
    return __async(this, null, function* () {
      const name2 = yield this.getNameFromURL(nameOrURL);
      yield this.deleteMany(
        (entry) => entry.name === nameOrURL || entry.name === name2
      );
    });
  }
  /**
   * Delete multiple files in cache.
   *
   * @param predicate A predicate like `array.filter(item => boolean)`
   */
  deleteMany(predicate) {
    return __async(this, null, function* () {
      const list = yield this.list();
      for (const item of list) {
        if (predicate(item)) {
          yield this.sb.delete(item.name);
          yield this.sb.delete(`${PREFIX_METADATA}${item.name}`);
        }
      }
    });
  }
  /**
   * Write the metadata of the file to disk.
   */
  writeMetadata(name, metadata) {
    return __async(this, null, function* () {
      const blob = new Blob([JSON.stringify(metadata)], { type: "text/plain" });
      yield this.sb.write(`${PREFIX_METADATA}${name}`, blob.stream());
    });
  }
};
var cache_manager_default = CacheManager;
function urlToFileName(url, prefix) {
  return __async(this, null, function* () {
    const hashBuffer = yield crypto.subtle.digest(
      "SHA-1",
      new TextEncoder().encode(url)
    );
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
    return `${prefix}${hashHex}_${url.split("/").pop()}`;
  });
}

// src/model-manager.ts
var DEFAULT_PARALLEL_DOWNLOADS = 3;
var ModelValidationStatus = /* @__PURE__ */ ((ModelValidationStatus2) => {
  ModelValidationStatus2["VALID"] = "valid";
  ModelValidationStatus2["INVALID"] = "invalid";
  ModelValidationStatus2["DELETED"] = "deleted";
  return ModelValidationStatus2;
})(ModelValidationStatus || {});
var Model = class {
  constructor(modelManager, url, mmprojUrl, savedFiles) {
    __publicField(this, "modelManager");
    /**
     * URL to the GGUF file (in case it contains multiple shards, the URL should point to the first shard)
     *
     * This URL will be used to identify the model in the cache. There can't be 2 models with the same URL.
     */
    __publicField(this, "url");
    /**
     * URL to mmproj file, if exists
     */
    __publicField(this, "mmprojUrl");
    /**
     * Size in bytes (total size of all shards).
     *
     * A value of -1 means the model is deleted from the cache. You must call `ModelManager.downloadModel` to re-download the model.
     */
    __publicField(this, "size");
    /**
     * List of all shards in the cache, sorted by original URL (ascending order)
     */
    __publicField(this, "files");
    this.modelManager = modelManager;
    this.url = url;
    this.mmprojUrl = mmprojUrl;
    if (savedFiles) {
      this.files = this.getAllFiles(savedFiles);
      this.size = sumArr(this.files.map((f) => f.metadata.originalSize));
    } else {
      this.files = [];
      this.size = 0;
    }
  }
  /**
   * Open and get a list of all shards as Blobs
   */
  open() {
    return __async(this, null, function* () {
      if (this.size === -1) {
        throw new WllamaError(
          `Model is deleted from the cache; Call ModelManager.downloadModel to re-download the model`,
          "load_error"
        );
      }
      const blobs = [];
      for (const file of this.files) {
        const blob = yield this.modelManager.cacheManager.open(file.name);
        if (!blob) {
          throw new Error(
            `Failed to open file ${file.name}; Hint: the model may be invalid, please refresh it`
          );
        }
        blobs.push(blob);
      }
      return blobs;
    });
  }
  /**
   * Validate the model files.
   *
   * If the model is invalid, the model manager will not be able to use it. You must call `refresh` to re-download the model.
   *
   * Cases that model is invalid:
   * - The model is deleted from the cache
   * - The model files are missing (or the download is interrupted)
   */
  validate() {
    let nbShards = ModelManager.parseModelUrl(this.url).length;
    if (this.mmprojUrl) {
      nbShards += 1;
    }
    if (this.size === -1) {
      return "deleted" /* DELETED */;
    }
    if (this.size < 16 || this.files.length !== nbShards) {
      return "invalid" /* INVALID */;
    }
    for (const file of this.files) {
      if (!file.metadata || file.metadata.originalSize !== file.size) {
        return "invalid" /* INVALID */;
      }
    }
    return "valid" /* VALID */;
  }
  /**
   * In case the model is invalid, call this function to re-download the model
   */
  refresh() {
    return __async(this, arguments, function* (options = {}) {
      var _a;
      const urls = ModelManager.parseModelUrl(this.url);
      if (this.mmprojUrl) {
        urls.push(this.mmprojUrl);
      }
      const works = urls.map((url, index) => ({
        url,
        index
      }));
      this.modelManager.logger.debug("Downloading model files:", urls);
      const nParallel = (_a = this.modelManager.params.parallelDownloads) != null ? _a : DEFAULT_PARALLEL_DOWNLOADS;
      const totalSize = yield this.getTotalDownloadSize(urls);
      const loadedSize = [];
      const worker = () => __async(this, null, function* () {
        while (works.length > 0) {
          const w = works.shift();
          if (!w) break;
          yield this.modelManager.cacheManager.download(w.url, __spreadProps(__spreadValues({}, options), {
            metadataAdditional: {
              originalURL: w.url,
              mmprojURL: this.mmprojUrl
            },
            progressCallback: ({ loaded }) => {
              var _a2;
              loadedSize[w.index] = loaded;
              (_a2 = options.progressCallback) == null ? void 0 : _a2.call(options, {
                loaded: sumArr(loadedSize),
                total: totalSize
              });
            }
          }));
        }
      });
      const promises = [];
      for (let i = 0; i < nParallel; i++) {
        promises.push(worker());
        loadedSize.push(0);
      }
      yield Promise.all(promises);
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      this.size = this.files.reduce((acc, f) => acc + f.metadata.originalSize, 0);
    });
  }
  /**
   * Remove the model from the cache
   */
  remove() {
    return __async(this, null, function* () {
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      yield this.modelManager.cacheManager.deleteMany(
        (f) => !!this.files.find((file) => file.name === f.name)
      );
      this.size = -1;
    });
  }
  getAllFiles(savedFiles) {
    const allUrls = new Set(ModelManager.parseModelUrl(this.url));
    if (this.mmprojUrl) {
      allUrls.add(this.mmprojUrl);
    }
    const allFiles = [];
    for (const url of allUrls) {
      const file = savedFiles.find((f) => f.metadata.originalURL === url);
      if (!file) {
        throw new Error(`Model file not found: ${url}`);
      }
      allFiles.push(file);
    }
    allFiles.sort(
      (a, b) => a.metadata.originalURL.localeCompare(b.metadata.originalURL)
    );
    return allFiles;
  }
  getTotalDownloadSize(urls) {
    return __async(this, null, function* () {
      const responses = yield Promise.all(
        urls.map((url) => fetch(url, { method: "HEAD" }))
      );
      const sizes = responses.map(
        (res) => Number(res.headers.get("content-length") || "0")
      );
      return sumArr(sizes);
    });
  }
};
var ModelManager = class _ModelManager {
  constructor(params = {}) {
    // The CacheManager singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "params");
    __publicField(this, "logger");
    this.cacheManager = params.cacheManager || new cache_manager_default();
    this.params = params;
    this.logger = params.logger || console;
  }
  /**
   * Parses a model URL and returns an array of URLs based on the following patterns:
   * - If the input URL is an array, it returns the array itself.
   * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
   * - Otherwise, it returns an array containing the input URL as a single element array.
   * @param modelUrl URL or list of URLs
   */
  static parseModelUrl(modelUrl) {
    var _a;
    if (Array.isArray(modelUrl)) {
      return modelUrl;
    }
    const urlPartsRegex = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
    const queryMatch = modelUrl.match(/\.gguf(\?.*)?$/);
    const queryParams = (_a = queryMatch == null ? void 0 : queryMatch[1]) != null ? _a : "";
    const matches = modelUrl.match(urlPartsRegex);
    if (!matches) {
      return [modelUrl];
    }
    const baseURL = modelUrl.replace(urlPartsRegex, "");
    const total = matches[2];
    const paddedShardIds = Array.from(
      { length: Number(total) },
      (_, index) => (index + 1).toString().padStart(5, "0")
    );
    return paddedShardIds.map(
      (current) => `${baseURL}-${current}-of-${total}.gguf${queryParams}`
    );
  }
  /**
   * Get all models in the cache
   */
  getModels() {
    return __async(this, arguments, function* (opts = {}) {
      const cachedFiles = yield this.cacheManager.list();
      let models = [];
      for (const file of cachedFiles) {
        if (!file.metadata.originalURL) continue;
        const shards = _ModelManager.parseModelUrl(file.metadata.originalURL);
        const mmprojUrl = file.metadata.mmprojURL;
        const isFirstShard = shards.length === 1 || shards[0] === file.metadata.originalURL;
        if (isFirstShard) {
          models.push(
            new Model(this, file.metadata.originalURL, mmprojUrl, cachedFiles)
          );
        }
      }
      if (!opts.includeInvalid) {
        models = models.filter(
          (m) => m.validate() === "valid" /* VALID */
        );
      }
      return models;
    });
  }
  /**
   * Download a model from the given URL.
   *
   * The URL must end with `.gguf`
   */
  downloadModel(_0) {
    return __async(this, arguments, function* (sourceOrURL, options = {}) {
      const source = isString(sourceOrURL) ? { url: sourceOrURL } : sourceOrURL;
      if (!isValidGgufFile(source.url)) {
        throw new WllamaError(
          `Invalid model URL: ${source.url}; URL must ends with ".gguf"`,
          "download_error"
        );
      }
      const model = new Model(this, source.url, source.mmprojUrl);
      const validity = model.validate();
      if (validity !== "valid" /* VALID */) {
        yield model.refresh(options);
      }
      return model;
    });
  }
  /**
   * Get a model from the cache or download it if it's not available.
   */
  getModelOrDownload(_0) {
    return __async(this, arguments, function* (source, options = {}) {
      var _a;
      const models = yield this.getModels();
      const model = models.find((m) => m.url === source.url);
      if (model) {
        (_a = options.progressCallback) == null ? void 0 : _a.call(options, { loaded: model.size, total: model.size });
        return model;
      }
      return this.downloadModel(source, options);
    });
  }
  /**
   * Remove all models from the cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.cacheManager.clear();
    });
  }
};

// src/types/types.ts
var LogLevel = /* @__PURE__ */ ((LogLevel2) => {
  LogLevel2[LogLevel2["DEBUG"] = 1] = "DEBUG";
  LogLevel2[LogLevel2["INFO"] = 2] = "INFO";
  LogLevel2[LogLevel2["WARN"] = 3] = "WARN";
  LogLevel2[LogLevel2["ERROR"] = 4] = "ERROR";
  return LogLevel2;
})(LogLevel || {});

// src/wasm-from-cdn.ts
var WasmCompatFromCDN = {
  worker: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.js",
  wasm: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.wasm"
};

// src/wllama.ts
var LoggerWithoutDebug = __spreadProps(__spreadValues({}, console), {
  debug: () => {
  }
});
var WllamaError = class extends Error {
  constructor(message, type = "unknown_error") {
    super(message);
    __publicField(this, "type");
    this.type = type;
  }
};
var WllamaAbortError = class extends Error {
  constructor() {
    super("Operation aborted");
    __publicField(this, "name", "AbortError");
  }
};
var WllamaRuntimeError = class extends Error {
  constructor(message, stack) {
    super(message);
    __publicField(this, "name", "RuntimeError");
    __publicField(this, "stack");
    this.stack = stack;
  }
};
var Wllama = class {
  constructor(pathConfig, wllamaConfig = {}) {
    // The CacheManager and ModelManager are singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "modelManager");
    /** Feature detection: `loraBlobs` means `lora_adapters` accepts GGUF blobs, `nativeStates` means `native_states` loads natlang native state spans, `natlangScore` means `createNatlangScore` is available, `slotState` means `saveSlot` / `restoreSlot` / `eraseSlot` / `applyTemplate` are available and completions accept `id_slot`, `nativeStateStore` means `putNativeState` / `getNativeState` are available. */
    __publicField(this, "capabilities", {
      loraBlobs: true,
      nativeStates: true,
      natlangScore: true,
      slotState: true,
      nativeStateStore: true
    });
    __publicField(this, "slotFileId", 0);
    __publicField(this, "compat", null);
    __publicField(this, "proxy", null);
    __publicField(this, "config");
    __publicField(this, "pathConfig");
    __publicField(this, "useMultiThread", false);
    __publicField(this, "nbThreads", 1);
    __publicField(this, "useEmbeddings", false);
    __publicField(this, "useRerank", false);
    // available when loaded
    __publicField(this, "loadedContextInfo", null);
    __publicField(this, "seed");
    __publicField(this, "bosToken", -1);
    __publicField(this, "eosToken", -1);
    __publicField(this, "eotToken", -1);
    __publicField(this, "eogTokens", /* @__PURE__ */ new Set());
    __publicField(this, "addBosToken", false);
    __publicField(this, "addEosToken", false);
    __publicField(this, "mediaMarker");
    __publicField(this, "chatTemplate");
    __publicField(this, "metadata");
    __publicField(this, "hasEncoder", false);
    __publicField(this, "decoderStartToken", -1);
    // the defaults are also the server's own (default_template_kwargs_keys/vals at load, as JSON texts), which is what
    // scoring renders with; the overlay keeps them on chat requests, whose own kwargs win
    __publicField(this, "chatTemplateKwargs", {});
    var _a, _b, _c;
    checkEnvironmentCompatible();
    if (!pathConfig) throw new WllamaError("AssetsPathConfig is required");
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
    this.cacheManager = (_a = wllamaConfig.cacheManager) != null ? _a : new cache_manager_default();
    this.modelManager = (_c = wllamaConfig.modelManager) != null ? _c : new ModelManager({
      cacheManager: this.cacheManager,
      logger: (_b = wllamaConfig.logger) != null ? _b : console,
      parallelDownloads: wllamaConfig.parallelDownloads,
      allowOffline: wllamaConfig.allowOffline
    });
    this.setCompat("default");
  }
  logger() {
    var _a;
    return (_a = this.config.logger) != null ? _a : console;
  }
  checkModelLoaded() {
    if (!this.isModelLoaded()) {
      throw new WllamaError(
        "loadModel() is not yet called",
        "model_not_loaded"
      );
    }
  }
  /**
   * Get the libllama version string, e.g. "b6327-4d74393".
   *
   * @returns version string embedded at build time.
   */
  static getLibllamaVersion() {
    return LIBLLAMA_VERSION;
  }
  /**
   * Set compatibility options for Wllama.
   * @param compat Set to null to disable compatibility, or 'default' to use the default compat resources from CDN.
   * @param mode 'safari' by default; If set to 'firefox_safari', the compat mode will **also** be enabled on Firefox, which will significantly degrade the performance but allow using WebGPU on Firefox.
   */
  setCompat(compat, mode = "safari") {
    if (mode === "safari") {
      if (isFirefox()) {
        this.compat = null;
        return;
      }
    }
    this.compat = compat === "default" ? WasmCompatFromCDN : compat;
  }
  /**
   * Check if the model is loaded via `loadModel()`
   */
  isModelLoaded() {
    return !!this.proxy && !!this.metadata;
  }
  /**
   * Get token ID associated to BOS (begin of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getBOS() {
    return this.bosToken;
  }
  /**
   * Get token ID associated to EOS (end of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOS() {
    return this.eosToken;
  }
  /**
   * Get token ID associated to EOT (end of turn) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOT() {
    return this.eotToken;
  }
  /**
   * Check if a given token is end-of-generation token (e.g. EOS, EOT, etc.)
   *
   * @param token the token ID to be checked
   * @returns true if the token is EOS, EOT, or any other end-of-generation tokens
   */
  isTokenEOG(token) {
    return token === this.eosToken || token === this.eotToken || this.eogTokens.has(token);
  }
  /**
   * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getDecoderStartToken() {
    return this.decoderStartToken;
  }
  /**
   * Get model hyper-parameters and metadata
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns ModelMetadata
   */
  getModelMetadata() {
    this.checkModelLoaded();
    return this.metadata;
  }
  /**
   * Check if we're currently using multi-thread build.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isMultithread() {
    this.checkModelLoaded();
    return this.useMultiThread;
  }
  /**
   * Get number of threads used in the current context.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns number of threads
   */
  getNumThreads() {
    this.checkModelLoaded();
    return this.useMultiThread ? this.nbThreads : 1;
  }
  /**
   * Check if the current model uses encoder-decoder architecture
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isEncoderDecoderArchitecture() {
    this.checkModelLoaded();
    return this.hasEncoder;
  }
  /**
   * Must we add BOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if BOS token must be added to the sequence
   */
  mustAddBosToken() {
    this.checkModelLoaded();
    return this.addBosToken;
  }
  /**
   * Must we add EOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if EOS token must be added to the sequence
   */
  mustAddEosToken() {
    this.checkModelLoaded();
    return this.addEosToken;
  }
  /**
   * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns the jinja template. null if there is no template in gguf
   */
  getChatTemplate() {
    var _a;
    this.checkModelLoaded();
    return (_a = this.chatTemplate) != null ? _a : null;
  }
  /**
   * Check if WebGPU is supported by the current environment.
   * @returns true if WebGPU is supported
   */
  isSupportWebGPU() {
    return isSupportWebGPU();
  }
  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelSourceOrURL
   * @param params
   */
  loadModelFromUrl(_0) {
    return __async(this, arguments, function* (modelSourceOrURL, params = {}) {
      var _a;
      const source = isString(modelSourceOrURL) ? { url: modelSourceOrURL } : modelSourceOrURL;
      const useCache = (_a = params.useCache) != null ? _a : true;
      const model = useCache ? yield this.modelManager.getModelOrDownload(source, params) : yield this.modelManager.downloadModel(source, params);
      const blobs = yield model.open();
      return yield this.loadModel(blobs, params);
    });
  }
  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param hfOptions
   * @param params
   */
  loadModelFromHF(_0) {
    return __async(this, arguments, function* (hfOptions, params = {}) {
      const source = yield getHFModelSource(hfOptions);
      return yield this.loadModelFromUrl(source, params);
    });
  }
  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param params LoadModelParams
   */
  loadModel(_0) {
    return __async(this, arguments, function* (ggufBlobsOrModel, params = {}) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k;
      const blobs = ggufBlobsOrModel instanceof Model ? yield ggufBlobsOrModel.open() : [...ggufBlobsOrModel];
      if (blobs.some((b) => b.size === 0)) {
        throw new WllamaError(
          "Input model (or splits) must be non-empty Blob or File",
          "load_error"
        );
      }
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (this.proxy) {
        throw new WllamaError("Module is already initialized", "load_error");
      }
      const supportMultiThread = yield isSupportMultiThread();
      const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
      const nbThreads = (_a = params.n_threads) != null ? _a : hwConccurency;
      this.nbThreads = nbThreads;
      this.useMultiThread = supportMultiThread && nbThreads > 1;
      const workerResources = this.getWorkerResources();
      if (params.n_gpu_layers === 0) {
        workerResources.noWebGPU = true;
      }
      this.proxy = new ProxyToWorker(
        workerResources,
        this.useMultiThread ? nbThreads : 0,
        // 0 means disable pthread
        (_b = this.config.suppressNativeLog) != null ? _b : false,
        this.logger()
      );
      let logLevel = (_c = params.log_level) != null ? _c : 2 /* INFO */;
      if (this.config.suppressNativeLog) {
        logLevel = 9999;
      }
      const modelFiles = yield prepareBlobs(blobs);
      const loraPaths = [];
      const sideFiles = [];
      for (const [i, adapter] of ((_d = params.lora_adapters) != null ? _d : []).entries()) {
        if (adapter.blob) {
          if (adapter.blob.size === 0) {
            throw new WllamaError(
              `LoRA adapter ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `lora-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: adapter.blob });
          loraPaths.push(`models/${name}`);
        } else if (adapter.path) {
          loraPaths.push(adapter.path);
        } else {
          throw new WllamaError(
            `LoRA adapter ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      const statePaths = [];
      for (const [i, state] of ((_e = params.native_states) != null ? _e : []).entries()) {
        if (state.blob) {
          if (state.blob.size === 0) {
            throw new WllamaError(
              `Native state ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `native-state-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: state.blob });
          statePaths.push(`models/${name}`);
        } else if (state.path) {
          statePaths.push(state.path);
        } else {
          throw new WllamaError(
            `Native state ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      yield this.proxy.moduleInit([...modelFiles.all, ...sideFiles]);
      this.logger().debug("Calling wllamaStart...");
      const startResult = yield this.proxy.wllamaStart();
      if (!startResult.success) {
        throw new WllamaError(
          `Error while calling start function, result = ${startResult}`
        );
      }
      this.logger().debug("Loading model...");
      const loadResult = yield this.proxy.wllamaAction("load", {
        _name: "load_req",
        log_level: logLevel,
        // if async read is not supported, use mmap; refer to README-dev.md for more details
        use_mmap: !canUseAsyncFileRead(workerResources.compat),
        use_mlock: false,
        n_gpu_layers: (_f = params.n_gpu_layers) != null ? _f : 99999,
        n_ctx: (_g = params.n_ctx) != null ? _g : 1024,
        n_threads: this.useMultiThread ? nbThreads : 1,
        n_ctx_auto: false,
        // not supported for now
        mmproj_path: modelFiles.mmproj ? `/models/${MMPROJ_FILE_NAME}` : void 0,
        model_paths: modelFiles.llm.map((f) => `models/${f.name}`),
        embeddings: params.embeddings,
        offload_kqv: params.offload_kqv,
        n_batch: params.n_batch,
        n_ubatch: params.n_ubatch,
        pooling_type: params.pooling_type,
        rope_scaling_type: params.rope_scaling_type,
        rope_freq_base: params.rope_freq_base,
        rope_freq_scale: params.rope_freq_scale,
        yarn_ext_factor: params.yarn_ext_factor,
        yarn_attn_factor: params.yarn_attn_factor,
        yarn_beta_fast: params.yarn_beta_fast,
        yarn_beta_slow: params.yarn_beta_slow,
        yarn_orig_ctx: params.yarn_orig_ctx,
        cache_type_k: params.cache_type_k,
        cache_type_v: params.cache_type_v,
        // with unified KV, all sequences share one n_ctx cache, so each request can still use the full context
        n_parallel: (_h = params.n_parallel) != null ? _h : 4,
        kv_unified: (_i = params.kv_unified) != null ? _i : true,
        flash_attn: params.flash_attn,
        swa_full: params.swa_full,
        chat_template: params.chat_template,
        jinja: params.jinja,
        reasoning: params.reasoning,
        image_min_tokens: params.image_min_tokens,
        image_max_tokens: params.image_max_tokens,
        warmup: params.warmup,
        no_kv_offload: params.no_kv_offload,
        mmproj_offload: params.mmproj_offload,
        cont_batching: params.cont_batching,
        n_keep: params.n_keep,
        ctx_shift: params.ctx_shift,
        cache_idle_slots: params.cache_idle_slots,
        n_cache_reuse: params.n_cache_reuse,
        lora_paths: params.lora_adapters ? loraPaths : void 0,
        lora_scales: (_j = params.lora_adapters) == null ? void 0 : _j.map((a) => {
          var _a2;
          return (_a2 = a.scale) != null ? _a2 : 1;
        }),
        lora_init_without_apply: params.lora_init_without_apply,
        native_state_paths: params.native_states ? statePaths : void 0,
        spec_draft_model: params.spec_draft_model,
        spec_draft_ngl: params.spec_draft_ngl,
        spec_draft_n_max: params.spec_draft_n_max,
        spec_draft_n_min: params.spec_draft_n_min,
        spec_draft_p_min: params.spec_draft_p_min,
        spec_draft_threads: params.spec_draft_threads,
        spec_draft_threads_batch: params.spec_draft_threads_batch,
        kv_overrides_keys: params.kv_overrides ? Object.keys(params.kv_overrides) : void 0,
        kv_overrides_vals: params.kv_overrides ? Object.values(params.kv_overrides) : void 0,
        // the server's own defaults too, so every request it renders gets them (natlang's /v1/natlang/score items as
        // well as chat), as llama-server's --chat-template-kwargs does; each value travels as its JSON text, which is
        // how the server stores them
        default_template_kwargs_keys: params.default_template_kwargs ? Object.keys(params.default_template_kwargs) : void 0,
        default_template_kwargs_vals: params.default_template_kwargs ? Object.values(params.default_template_kwargs).map((value) => JSON.stringify(value)) : void 0,
        reasoning_budget_tokens: params.reasoning_budget_tokens,
        reasoning_budget_message: params.reasoning_budget_message,
        reasoning_format: params.reasoning_format,
        skip_chat_parsing: params.skip_chat_parsing,
        prefill_assistant: params.prefill_assistant
      });
      const loadedCtxInfo = __spreadProps(__spreadValues({}, loadResult), {
        metadata: {}
      });
      for (let i = 0; i < loadResult.metadata_key.length; i++) {
        loadedCtxInfo.metadata[loadResult.metadata_key[i]] = loadResult.metadata_val[i];
      }
      this.seed = params.seed;
      this.bosToken = loadedCtxInfo.token_bos;
      this.eosToken = loadedCtxInfo.token_eos;
      this.eotToken = loadedCtxInfo.token_eot;
      this.useEmbeddings = !!params.embeddings;
      this.useRerank = params.pooling_type == "rank";
      this.metadata = {
        hparams: {
          nVocab: loadedCtxInfo.n_vocab,
          nCtxTrain: loadedCtxInfo.n_ctx_train,
          nEmbd: loadedCtxInfo.n_embd,
          nLayer: loadedCtxInfo.n_layer
        },
        meta: loadedCtxInfo.metadata
      };
      this.hasEncoder = !!loadedCtxInfo.has_encoder;
      this.decoderStartToken = loadedCtxInfo.token_decoder_start;
      this.addBosToken = loadedCtxInfo.add_bos_token;
      this.addEosToken = loadedCtxInfo.add_eos_token;
      this.chatTemplate = loadedCtxInfo.metadata["tokenizer.chat_template"];
      this.loadedContextInfo = loadedCtxInfo;
      this.eogTokens = new Set(loadedCtxInfo.list_tokens_eog);
      this.mediaMarker = loadedCtxInfo.media_marker;
      this.chatTemplateKwargs = (_k = params.default_template_kwargs) != null ? _k : {};
      this.logger().debug({ loadedCtxInfo });
    });
  }
  getLoadedContextInfo() {
    this.checkModelLoaded();
    if (!this.loadedContextInfo) {
      throw new WllamaError("Loaded context info is not available");
    }
    return __spreadValues({}, this.loadedContextInfo);
  }
  //////////////////////////////////////////////
  // High level API
  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param options OAI-compatible embedding creation options
   * @returns OAI-compatible embedding response
   */
  createEmbedding(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      if (!this.useEmbeddings) {
        throw new WllamaError(
          "Embeddings is not enabled. Please set it via LoadModelParams.embeddings"
        );
      }
      const result = yield this.proxy.wllamaAction(
        "embedding",
        {
          _name: "embd_req",
          data_json: JSON.stringify(options),
          files: []
          // TODO: support file input
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(options, false, result.req_id);
    });
  }
  /**
   * Rerank a list of documents against a query.
   * Requires the model to be loaded with embeddings: true and pooling_type: 'rank'.
   * @param options Reranking options (query, documents, top_n)
   * @returns Reranking response with relevance scores sorted highest first
   */
  createRerank(options) {
    return __async(this, null, function* () {
      var _a, _b;
      this.checkModelLoaded();
      if (!this.useEmbeddings || !this.useRerank) {
        throw new WllamaError(
          "Rerank is not enabled. Please set it via LoadModelParams: embeddings = true and pooling_type = rank"
        );
      }
      const top_n = (_a = options.top_n) != null ? _a : options.documents.length;
      let totalTokens = 0;
      const rawResults = [];
      for (let i = 0; i < options.documents.length; i++) {
        const result = yield this.proxy.wllamaAction("rerank", {
          _name: "rrnk_req",
          data_json: JSON.stringify({
            query: options.query,
            document: options.documents[i]
          })
        });
        if (!result.success) {
          throw new WllamaError(
            "Model failed to start reranking",
            "inference_error"
          );
        }
        const { score, tokens_evaluated } = yield this.getSingleResult(result.req_id, "reranking");
        totalTokens += tokens_evaluated;
        rawResults.push({ index: i, score });
      }
      rawResults.sort((a, b) => b.score - a.score);
      return {
        model: (_b = this.getModelMetadata().meta["general.name"]) != null ? _b : "",
        object: "list",
        usage: { prompt_tokens: totalTokens, total_tokens: totalTokens },
        results: rawResults.slice(0, top_n).map(({ index, score }) => ({
          index,
          relevance_score: score
        }))
      };
    });
  }
  /**
   * Answer typed questions about a state, using a decision model (TypeSafe-compatible System One API).
   * Image input is not supported yet.
   * @param options The state and the questions
   * @returns The answer of each question
   */
  createSystemOne(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "systemone",
        {
          _name: "sys1_req",
          data_json: JSON.stringify(options)
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start systemone",
          "inference_error"
        );
      }
      return yield this.getSingleResult(
        result.req_id,
        "systemone"
      );
    });
  }
  /**
   * Score candidate replies to chat messages (natlang decision readout): the log-probability of each reply as the closed final assistant message, summed over its tokens after the chat prompt. The replies of an item share the prompt.
   * @param options The items to score
   * @returns Per item, the log-probability and token count of each reply, or the error of an item that could not be rendered
   */
  createNatlangScore(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("score", {
        _name: "scor_req",
        data_json: JSON.stringify(options)
      });
      if (!result.success) {
        throw new WllamaError("Model failed to start scoring", "inference_error");
      }
      return yield this.getSingleResult(
        result.req_id,
        "score"
      );
    });
  }
  /**
   * Render chat messages (and tools) with the chat template, as the server's POST /apply-template does.
   * @param options OAI-compatible chat completion options (messages, tools, add_generation_prompt, ...)
   * @returns The prompt text
   */
  applyTemplate(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "apply_template",
        { _name: "tmpl_req", data_json: JSON.stringify(options) }
      );
      if (!result.success) {
        throw new WllamaError("Failed to apply chat template", "inference_error");
      }
      return result.prompt;
    });
  }
  /**
   * Save the state of a slot (its KV cache, tokens and context checkpoints), as the server's POST /slots/:id?action=save does. The slot must be idle.
   * @param idSlot The slot
   * @returns The saved state, to give to `restoreSlot` of an engine with the same model, build and context settings
   */
  saveSlot(idSlot) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const res = yield this.slotAction(
        idSlot,
        "save",
        filename
      );
      const state = yield this.proxy.slotFileRead(filename);
      return { state, n_saved: res.n_saved, n_written: res.n_written };
    });
  }
  /**
   * Restore a slot from a state of `saveSlot`, as the server's POST /slots/:id?action=restore does. A state that does not fit leaves the slot empty and throws.
   * @param idSlot The slot
   * @param state The saved state
   * @returns The number of restored tokens and bytes read
   */
  restoreSlot(idSlot, state) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        return yield this.slotAction(idSlot, "restore", filename);
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Clear the cached prompt of a slot, as the server's POST /slots/:id?action=erase does.
   * @param idSlot The slot
   */
  eraseSlot(idSlot) {
    return __async(this, null, function* () {
      return yield this.slotAction(idSlot, "erase", "");
    });
  }
  slotAction(idSlot, action, filename) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("slots", {
        _name: "slot_req",
        id_slot: idSlot,
        action,
        filename
      });
      if (!result.success) {
        throw new WllamaError(`Failed to ${action} slot`, "inference_error");
      }
      return yield this.getSingleResult(result.req_id, `slot ${action}`);
    });
  }
  /**
   * Store a natlang native state (natlang's store-file format, natlang.native-state/1) after the model is loaded, as the
   * server's PUT /v1/neuralese/native-states/:id does: the bytes are verified (layout, finite values, content ID), kept as
   * received and, for a state of this model's identity, registered so prompts consume it exactly.
   * @param state The state's bytes
   * @param id The ID the state must have (its content ID); omit to accept any
   * @returns The consumption mode and the state's metadata ({id, ...metadata})
   */
  putNativeState(state, id = "") {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const filename = `wllama-native-state-${++this.slotFileId}.nzs`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        const result = yield this.proxy.wllamaAction(
          "native_state",
          { _name: "nsta_req", action: "put", id, filename }
        );
        if (!result.success) {
          throw new WllamaError("Failed to put native state", "inference_error");
        }
        return yield this.getSingleResult(result.req_id, "native state put");
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Read a stored native state (one this engine produced, or one put or loaded), as the server's
   * GET /v1/neuralese/native-states/:id does: its bytes exactly as stored, natlang's store-file format.
   * @param id The state's ID
   * @returns The state's bytes and its metadata ({id, ...metadata})
   */
  getNativeState(id) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const filename = `wllama-native-state-${++this.slotFileId}.nzs`;
      const result = yield this.proxy.wllamaAction(
        "native_state",
        { _name: "nsta_req", action: "get", id, filename }
      );
      if (!result.success) {
        throw new WllamaError("Failed to get native state", "inference_error");
      }
      const state = yield this.proxy.slotFileRead(filename);
      return { state, meta: JSON.parse(result.meta_json) };
    });
  }
  createChatCompletion(options) {
    return __async(this, null, function* () {
      var _a;
      if (Object.keys(this.chatTemplateKwargs).length > 0) {
        options = __spreadProps(__spreadValues({}, options), {
          chat_template_kwargs: __spreadValues(__spreadValues({}, this.chatTemplateKwargs), (_a = options.chat_template_kwargs) != null ? _a : {})
        });
      }
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  createCompletion(options) {
    return __async(this, null, function* () {
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  /**
   * Private implementation of createCompletion
   */
  createCompletionImpl(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const isStream = !!options.stream;
      const isChat = !!options.messages;
      const customOpt = {};
      if (this.seed !== void 0) {
        customOpt.seed = this.seed;
      }
      let files = [];
      if (isChat) {
        const tmp = this.prepareMultimodalInput(
          options
        );
        options = tmp.params;
        files = tmp.files;
      }
      const result = yield this.proxy.wllamaAction(
        "completion",
        {
          _name: "cmpl_req",
          is_chat: isChat,
          data_json: JSON.stringify(__spreadValues(__spreadValues({}, options), customOpt)),
          files: files.map((f) => new Uint8Array(f))
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(
        options,
        isStream,
        result.req_id
      );
    });
  }
  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   * Only called when stream=true and no onData is provided.
   */
  createCompletionGenerator(options) {
    return new Promise((resolve) => {
      const createGenerator = cbToAsyncIter(
        (callback) => {
          this.createCompletionImpl(__spreadProps(__spreadValues({}, options), {
            onData: (chunk) => callback(chunk)
          })).then(() => callback(void 0, true)).catch((err) => callback(void 0, false, err));
        }
      );
      resolve(createGenerator());
    });
  }
  /**
   * Whether the currently loaded model supports a specific input modality (e.g. image or audio).
   * @param modality
   * @returns
   */
  supportInputModality(modality) {
    this.checkModelLoaded();
    if (modality === "image") {
      return !!this.loadedContextInfo.has_image_input;
    } else if (modality === "audio") {
      return !!this.loadedContextInfo.has_audio_input;
    } else {
      throw new WllamaError(
        "Unsupported modality: " + modality,
        "unknown_error"
      );
    }
  }
  /**
   * Unload the model and free all memory.
   *
   * Note: This function will NOT crash if model is not yet loaded
   */
  exit() {
    return __async(this, null, function* () {
      var _a;
      yield (_a = this.proxy) == null ? void 0 : _a.wllamaExit();
      this.proxy = null;
    });
  }
  /**
   * [FOR DEBUGGING ONLY] Run ggml backend ops tests without loading any model.
   *
   * Initializes the wasm runtime, executes `test-backend-ops` with the given args, then shuts down.
   *
   * For more info, please refer to guides/debug.md
   *
   * @param args Arguments forwarded to test-backend-ops (e.g. ["-o", "ADD"])
   * @returns retcode (0 = all tests passed) and success flag
   */
  testBackendOps() {
    return __async(this, arguments, function* (args = []) {
      var _a;
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (!(yield isSupportMultiThread())) {
        throw new WllamaError(
          "Multi-threading is required to run backend ops tests, but it is not supported in the current environment."
        );
      }
      const tmpProxy = new ProxyToWorker(
        this.getWorkerResources(),
        0,
        // single-thread; no model needed
        (_a = this.config.suppressNativeLog) != null ? _a : false,
        this.logger()
      );
      try {
        yield tmpProxy.moduleInit([]);
        const startResult = yield tmpProxy.wllamaStart();
        if (!startResult.success) {
          throw new WllamaError(
            `Error while calling start function, result = ${startResult}`
          );
        }
        const result = yield tmpProxy.wllamaAction(
          "test_backend_ops",
          { _name: "tbop_req", args: ["test-backend-ops", ...args] }
        );
        return { retcode: result.retcode, success: result.success };
      } finally {
        yield tmpProxy.wllamaExit();
      }
    });
  }
  //////////////////////////////////////////////
  // Low level API
  // TODO: add back
  /**
   * get debug info
   */
  _getDebugInfo() {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      return yield this.proxy.wllamaDebug();
    });
  }
  //////////////////////////////////////////////
  // Utils
  jsonDecode(data_json) {
    try {
      return JSON.parse(data_json);
    } catch (e) {
      this.logger().error("Failed to parse JSON:", data_json);
      throw new WllamaError("Failed to parse model output", "inference_error");
    }
  }
  prepareMultimodalInput(params) {
    const msg = params.messages;
    const msgNew = [];
    const files = [];
    for (const m of msg) {
      if (Array.isArray(m.content)) {
        const newContent = [];
        for (const c of m.content) {
          if (c.type === "text" || c.type === "native_state") {
            newContent.push(c);
          } else {
            if (!this.mediaMarker) {
              throw new WllamaError(
                "Media marker is undefined",
                "inference_error"
              );
            }
            files.push(c.data);
            newContent.push({
              type: "text",
              text: this.mediaMarker
            });
          }
        }
        msgNew.push(__spreadProps(__spreadValues({}, m), {
          content: newContent
        }));
      } else {
        msgNew.push(m);
      }
    }
    return {
      params: __spreadProps(__spreadValues({}, params), {
        messages: msgNew
      }),
      files
    };
  }
  // release the slot occupied by the request; cancelling an already-finished request is a no-op
  cancelRequest(reqId) {
    return __async(this, null, function* () {
      try {
        yield this.proxy.wllamaAction("cancel", {
          _name: "cncl_req",
          req_id: reqId
        });
      } catch (e) {
        this.logger().warn("Failed to cancel request", reqId, e);
      }
    });
  }
  getSingleResult(reqId, task) {
    return __async(this, null, function* () {
      let completed = false;
      try {
        while (true) {
          const chunk = yield this.proxy.wllamaAction(
            "get_result",
            { _name: "gres_req", req_id: reqId }
          );
          const jsonString = chunk.data_json;
          if (jsonString && jsonString.length > 0) {
            if (chunk.is_error) {
              const jsonData = this.jsonDecode(jsonString);
              throw new WllamaError(
                jsonData.message || `Unknown ${task} error`,
                "inference_error"
              );
            }
            completed = true;
            return this.jsonDecode(jsonString);
          }
          if (!chunk.has_more) {
            completed = true;
            break;
          }
        }
        throw new WllamaError(`No ${task} result received`, "inference_error");
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
    });
  }
  getResponse(options, isStream, reqId) {
    return __async(this, null, function* () {
      var _a, _b;
      let finalResult = null;
      let completed = false;
      try {
        while (true) {
          if ((_a = options.abortSignal) == null ? void 0 : _a.aborted) {
            throw new WllamaAbortError();
          }
          const result_chunk = yield this.proxy.wllamaAction(
            "get_result",
            {
              _name: "gres_req",
              req_id: reqId
            }
          );
          const jsonString = result_chunk.data_json;
          if (!jsonString || jsonString.length === 0) {
            if (!result_chunk.has_more) {
              completed = true;
              break;
            } else {
              continue;
            }
          }
          if (jsonString == "null") {
            continue;
          }
          let jsonData = this.jsonDecode(jsonString);
          finalResult = jsonData;
          if (result_chunk.is_error) {
            this.logger().error("Model returned an error:", jsonData);
            throw new WllamaError(
              jsonData.message || "Unknown inference error",
              "inference_error"
            );
          }
          if (isStream) {
            if (!Array.isArray(jsonData)) {
              jsonData = [jsonData];
            }
            for (const chunk of jsonData) {
              (_b = options.onData) == null ? void 0 : _b.call(options, chunk);
              finalResult = chunk;
            }
          }
          if (!result_chunk.has_more) {
            completed = true;
            break;
          }
        }
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
      return finalResult;
    });
  }
  getWorkerResources() {
    const workerResources = {
      wasmPath: absoluteUrl(this.pathConfig["default"]),
      compat: false
    };
    if (needCompat()) {
      if (!this.compat) {
        this.logger().warn(
          "Not using compat mode" + (isFirefox() ? " (expected on Firefox - WebGPU will be disabled)" : "")
        );
      } else {
        const isUsingDefault = this.compat.worker === WasmCompatFromCDN.worker && this.compat.wasm === WasmCompatFromCDN.wasm;
        if (isUsingDefault) {
          this.logger().warn(
            "Compatibility mode is activated, using resources from CDN. To use local resources, please refer to @wllama/wllama-compat package."
          );
          this.logger().warn(
            "IMPORTANT: Performance will be significantly degraded in compatibility mode."
          );
        }
        workerResources.wasmPath = absoluteUrl(this.compat.wasm);
        workerResources.jsPath = this.compat.worker;
        workerResources.compat = true;
      }
    }
    if (isFirefox()) {
      if (workerResources.compat) {
        this.logger().warn(
          'Using compat mode on Firefox, performance will be significantly degraded; Consider enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      } else if (!isSupportJSPI()) {
        this.logger().warn(
          'WebGPU is disabled on Firefox due to missing JSPI support. Please consider enabling compat mode, or enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      }
    }
    return workerResources;
  }
};
export {
  CacheManager,
  LogLevel,
  LoggerWithoutDebug,
  Model,
  ModelManager,
  ModelValidationStatus,
  POLYFILL_ETAG,
  Wllama,
  WllamaAbortError,
  WllamaError,
  WllamaRuntimeError,
  getHFFileSHA256,
  getHFModelSource,
  isValidGgufFile
};
