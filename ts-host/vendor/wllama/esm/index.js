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
  "compat": "H4sIAAAAAAAAA+S9a4/dRpI22ED7IrVt3VVVKt1Kd1KSZdXFandZVo/bdns8bk973JfBzLyDRB4yzzns4k1M8lSVsSjsALtf9uNigQX2L+1PWmCxiIhMMjOZ5Dmq9tvbwPtFqhMRmUySyczIuDzxf//sZz/7fx797Gf/tfazn62LTEZVUtYiZ5MmSeskZ9NKiF/MZlnK+KSo6ttpyjPOsiIWKZtwKfb3o0rwWrBa5LKo3i9KUfG6qLZycXg9KrKsyNlfZJHv758Yv4LwgfGLJbXIbAEkBeEGY9ERZzxNiwguIo4iUdZJkV8hBoyuI56Rdby//+zZszXGopTnMxbxNGW1qLIk57V4JxNZlJVX0zRjs4qXcxYVeS2O6v39aPIB3WQcsyQr07fEUTl9R9ZVKvKPJkWRbpk3nhY8FtX+/kzU7EAcv2xymcxyEW8lz549e/XIvI8FTxvh3BvSgvAy3cOs4VXMKpEKLoVF49HrJqnEeRxZJeScl4Ltxo+WjmeaFrx+dRbbwS2ei9KkpFvublWKmuU8E/ewO3x0+uV99tnLaM6rp0iFW6IWWZPiw1nHX4nEvpNZUzSSVcWhXLPed9vZ+23jjNe7SqYUM9bkdZKyklcSBn8ywAnCGwZHS+ven9hPZyc+x1h8nPMsiVjEZW2z92IaTCmqrKnFJXqIaTFjSV6LKufpe8jPJ8e1kOfUdYHP4/jzPC3mGc/z/X18jXySsMUu295hz/f3J1wmEd75y04MHp3Jwgk4r27QZeclr3gm9/dzJrJJzOaCx+zgkWd2nvRoQXi11wm03/BS2cHiMk3vSjDJM8HwcVzuCU+nb0WFnF7pMaqivmq9XRgHT3J53zPcHmnshhcXrSkPi8yjw1nZ7O//Jsnjr6uiKb/K6+p4f//EJgThdX+3CzZ7za/h/cIqFrNpUR1C5+Ko5Ll+RPjlwFjbHwPdHUB3e4siiekjYayGVWfCU55HgvFpLSqW5FJU9ctOAF49zbVFIg7ZTvzA/GBjESX4Wm1SEHqe1FEpqpo1UsTv6Q864vVbMsmnF5AwLaqM07e85Xkb9AzSouIsyy71P90rNOPFoVrC8RN/NymrJK+n9OHDugIf+Ll24UjysqnPyJyk3qmKJo+nX674geBTynj5dIs+kJOOF4Rr5lpFY8+LKjOe5W58zbNRZbhJ3FuyP7HXBwvazmTEU/HOTNQiX7zF6yLZyDiLKzZNecRgOS54zCIezcWZqXoYl6wvQORxENITqSuey7KQgn5WRSnwKd4cfB1wS9f+kuR/4fv70yaPGK9mkhbwspC3+hyRy6YSLCqavDaexV78jD6W30/+IqL6N1yKl0T4oshK9Xg7XhBect/2dvxWWsym70wPq6QWn5jzfM7lnNV8koqXHYX6ZCwuWCVA4mVdNeLVumctgbl771BMZmXDyqQUaZLDVuhQgvDXY/MmFjVP4EMVR6Lqra0zUQfhOdisXr3aiopc1g/D6/Tp8+hA5DGbNNMpfKCSzQtZ03yuMolv4ANZTxIG03v7RSp+qV5vNOc1y+SMxSJNsqQ2N6c+LwjvOby+dBA+Min48ThSSAvCF8v2dlZUjFcVvRBeVfwYHsQVz468fae3bdJ6CL1VohRcTSRow3YGJ9JvmunUM5HeV0+5qhJR0Q+ZzLIiiS/iD3NT/X+NkYhSJin8jSN6umWwZM2r2seA9+ghp/D8eertSrxuBCzPHl40LxI/B55KnYAq6eNy/yjywjtmnh9r8mVr3ZDJjyII3655Pp++m9LKqVQ1Luur/eVvOs2vw0JSFwciT34UFStTnhU7in+/098qMRNHoGVFc5og8GEwZyvaju+OTAz1TC/VkslmgjudUk3Nj2UirktRLUTFai4P9vdPjF8w2dt7YFExW6So3Lu0INxUyxzqwwy+CPVXEH4wmzVTNk3yGCY9KOJRVl7EPsyJ9etuuVqIqC6ql0lek9q6rxcqLkE5Z4dJPWfw7EHk8dMt+PfVaXesmNd8f/8E/gvCh0v1cRB49W5csLKpo384pRrZ6tPb+vXhUgv6XJkmubHsOIwgvGI96KjISl7pvQoXAranvtppubvD6oJNy+0XZxiT85moo7H5Ikseie44hzO/Ejm39XkkBeE194UnsA7DFmCr6dvxVfPMgBrHPMnrs2qZSZveNrYTt4vZgu3E17udtxKySev9fTzawSp7vtv/2SThcj2rM1BG01JUMLtm+B0UszXGynpegZIKh4Uj1uRpER1sWDPf+HHTUixJZyoFm3KYmFIdupoJagZX3L6h53Vr41I3JkUdmhuF3mBP+sQgPMPgmdXT3W01rkrIssglfMQ0JU/8DNRkmimrj0uBmiQ9zTrJBGvk1sgEEKUMwrMLrQqeVytZrmbWh2q3jxNZwrrEYiGjTgmwyEF4AQeBn80Cx3K+1TiLpi6b+lp/eUzyklX88P1DevyxmDSzhz11vRJZsRCOgn69rAQ7nMl0f/+ro7L6HagY+/s5nq/eKovD6S3PhE2yrEGlCFcpfKcZasc0J2UxrVnGj/A104QFVYO+tmmS0hH6ka1s8pjBVen3tIIT2hzfyx3fsmJptHskcbAgZZXFcrFn2DWKrITtQtlf6O8gXHO0tUSySkRqGy85HJXqS7/73Xfs2z93C08Q3rceB2of8FDgfSk1XAahOkGzOMmkPj6kDRPVlBTvKOVZeW9kRtH+HIRXrG0TrxaEOKo//jP75rvvf7e/L+uqPTO19yyP82heFTnus2MXynh1AFP/Dk2kRVKzoqzh27IJQfgc59OgVr7gKc0sOec0s675z5JSr3R6ZUu8Z5kIzzJPzC9cCikT04xmUoPwS3PCyzJNaqX46hkva07biMjKlEcClxmH+aTms5lQSk67bp54qEG4obSBqjjUKnaTSz4Va2qBkWlRg0aCukMQ0oJOKzLMF2Uu7BTQ3Xh0lYHD1ldvfJPTqshrhwvKRF3E1+gLPOQL+v7KKKMvT940v81YREUsjA/zPky1Oc9zAW+wZEme1NB1HoOJQXHOJJKJ1w1PN5TeJRNcpEVeS3oIV7qTIq5u06kU9Rn9/d2xdoJM1BwfGt2zrHktttxDTo0SOHFQzbnhOwVhJ8mP4ufiqPQctHld5+8ekpGqU/d2d1LRmQxhpaNxRuWx8f724qvGTJ7ipnbID8QOb+pCW+3Mv6MiTUVUs7ISOGViUnBlEP6C4XY4TVJx/Tf0SfzQpHB0NH4F4TtSRjyfbuP2nWR8JqCvsioi2Cz290/8jCBUG3UzgZ2nNcPQzyC82zchoC1AHnJW8hoUT9f2lkjgqsUubXCFf58pfQFu4zc9PRX/Rm1vUE+1RW5rTQMfkqFyq4f286Kpf54WM2XUWxQRn+DIRDFTb09mqB6dVbpW2lxtrTvTpJL6o9zwzRywtGslD7WCTHBDC2lJQThuLu+fXT9g7LDiJZtGaSGF0mCSXfhDZFF5fK6WejkA7eznJY8/tvc83AcEh7srj1mSTwt4PA6pO250ZtXWmHqlfRC4DdNz2FWKMlvwKuHw5bZL4gAnCG1VrSqy0j3oEzEI172aTBCehZkwhyew7rzxspA4vC2/bjsFEyHLi1jcGDQ7vT5Y3OxPb+g0hyVuJndnsLODgQNuMRV4Fq4EvOkBjt7wSd275ps78IHXb0+lEAdXRQZWhS9Qg2ntuWMrP7oC7tW4oteiU38dCqoBA3c9neZXcfSyrvSNoOZAXwKcxOn0U1YiwkWOR1HvhLEXb3q26krgkvshvk1aa6a7O0wvLT5yEG51R3bGZFHVu2qP+iIlA9v7sFtVoqqKiqWBo+F1io42xLIDlsRH8u6YX4FlKQ/CTZroOOaq86vBrKFFIk4WuHytWS8SF8CJkPUNXpadw21/3/p5hoGbpJ7uPoLjkJqctF2hGuHQgvDO4CuDnYhlcz6gR1VBSHuQOCrfx5NWDjYfEe8akxO/G3VuUN+shxOEY3Yyskm/xxi61GRST6/A/G1n7g+k3/zDCus7KgpDq/sUO1jFDiqOyPTB0NpZNXi1oG1JV2t5jhVhHcw6NY8O1HmILUQFCuT7qBMdse29j3d/dbb1rQ3br+H9hCvMy9cs4/LgVn+/TPIYTlts2qTplnvKquegVuJZFL+Bs+0x4rJ5yGdxUomo9h6bZ6K+Z9FlNBftiiSbsiyqWsRrfTUKTr/viKMy256+LY7KnenY/MjFjNdiTIJOM184pplSRE3K62QhWFzxad033rgSQXjePg/VN9Xvdq0xfgbh+3pbBe32/tdf/+m33wkwWn397Z9fdr9aKzo7WDyzDs34CsigiO/gZdCaGKFREK51Xu+87qzv5+sqydjhPKkF2oY+kHWV5DPlpbpunQ5oL2RRKngVhHtpkc/o+k2egDzY+cBEUFfJpIEv9yWIvDKPpE9oatEXDfY4dSwteVUnPMXVFSYjHAfG1scFrY/qvI3PkJZHtHGB2o27WBB6fDMn7d/ajKI0HZYri1UtSlp5Xzc8r5MfBR7+Xn/CnrO9o09oA02ynahIN3yTGZSYh/0jgZ7MsF5rc4DtpS3QYB+Efh//IZqHwV6q9hycMOiCpUNSSv9N3wVj2SSfbrw+FPmeOCpZJTA0Aax3P4qquIKq9kzkaI2JxZQ3aX2LzlFZuatPUWTjQPOGONqBxYVWk1ikNWc5mBFI3x7gPBtacfz0m0PiUXmMjnZHiUyLutUe9d9BeNsxL9C2wSpRVGg4u2i9cVi0yC27EBGLixr2fTULXlfKgpkJnu/DXzCUBr0xPDrY31evu654UsuXEzg9sNfb7Dl9pYdFdYB7xzvTadrI+c1YiBKUKrbYo3leF0Wqtlh5H1WPdq7DmoweK6WNkFJpWupnFc+TWuzhaRUmD70BHMVtv9mNdmkuDwz1RSbR5+ns+yJNouOnW6sYjM7ok9K7jMkkj6fw/zxNsncZiwoZT28azgjZZBmv4AuK5kkaVyJH58C8+kV7Xv1kzbLQthP+sra4SlHD8ppMk+iLVbZcKaqEp+B56Rnl4yYrr5r2Anwfk6SWw674iJd1U4mny1zVsLOwBvSrUj5aRQesilqdu5QRMCu19gUzXkZFJW6OCBwstIEeiZnIa9A3RNTU4kwiD3E9f5rktd4k4D02JfgLwNw3g0lrrylLhO0gI/tDgrX8jP5uhhfPPXvBjMWCTY7xK/QeRXCreU/FwvB8Ju72z7QuZSz86H/adZnWIdHHCcK71hqNDre6MDp9bOzeRSnyDxjYY0Uex9N6uvPuYVTVRTb5oNXlykbOL9GfoMhrh92Wbd3+Xg0dx/J5HAfhz8vi8OPVdM5W4T+c8zoIV/Wb9T4X028W5qTe4GSjdWZ//6RPDMKbhiVHn4bz1vKx3m4+vImTAmcQLN1bxndZzGYTydisgOkHu1DJZ+IGLesLiK/pzH9KU7k3+P3S0GTJc3U6QUcbWNcqIeJjVCAHtx5UtVSwBE5G/bXDMwl92sdJnxiEj9WHilER5oHsxKEE4f2ZWnt7Z5ITzQlCMs3XRckOPmjdGGXaKFNfWsxo54JDzM0BfZtxsLz/a19PsRQUpiLH0Bm1smwQXouT6ZQsoDCD2h9BOGY94PlxED4e+YBfvgw0Fz1CnRYuBWvq6ScMNJeySPL6tqWhuxp7F6ao9HFYmcnoD/Plqs1UKu9Dc1+tBGzuEP9DIQgim4g4TvKZfNDffslrlokKjJsUzfgPQ5uw2g4hwgx29m4y0l2+boSEC8rPVtrFSafsb+NrFIXTngSeopUjvG/aOqIiR0NqSXMRV2C0yTN2yJOabH5S1H17aRupmQzaS22Rz7sulJmj06w6qlqWhKyr4pjoJU+ql0C/a56HKgFHZeHYaR6YIqoXxuuX6omXVYGTCk0fneWq+xWEV0htKWWsleQqFRdaXYaRj+o9Wt0hKrr6uUzyDdc02Mg5fkcfKcasSWL0LNQwfzprsU0PwgeKbq1PPd/wO6T//6+nipZQ02HUuDESasFrVBOoFyCpzQB1nmje5Adkt1XP4oHLbf0AHSkIbww6f1gqrnXnFlbkpNFlIiuq42GdLisEmBnP0RzMWSUhBEnZPZKCYXAduH/B3MowujgI3Zi5OZfoEVLbCkmfdD+CcMs5i1RCFulCKNtvUUp1RR7zssZohorTOz0UyWxeryevdyiqBox5LAfipGgG9Ktv4XDas4Puxud0tBlMC66sM+AgsN1WlZjhq0HHC4/jSkh5o6+tddKDLivQjt8S1XRqBVhRmIl+BJWYwiPVXqVK8GgOXnpWNamQevml9wiL82tWZecYkxA3ylR8J72lQ77YfqH87++BFgZxIVFWDgXIfSkWiSdAbiQwM/OI95e8FUKZ1MpzWKFns98FWi2xk8EuUAS/LhDZ6JTKCo0vRU0T/l15LKMin6r1SqtNpCttOcsRvDP9pioxTY5CbfaBI0hrsOoTwa+siXORcfomcN956DCmiUBdDNSI9u8gvDui9P4p5xA2vuboizA/hZTnMn4g2OuSPgP5CK1R+I+lrs9E/bJlvUJn4tA5byGiZ8NGDqJWImoq8AvjYfoDEOdVNEcl8rxeTWC2Zvzohj4xCm2ygAdNAeLStcYm4K+0lnXQQxoZhNdcwwe+LpGm8razIIEmhrZWZa//aODweOKld3FnES9RX+IQK0THvIMFXTdvYwPAXtIGz0x3d9ThL8k/0MuNyMr6+KqKU9t+QXFquzvgkj5v24UPjWCwbcfIi0GQg3qs1HqsbVWGocZikURiZbvNHvvWsds8VRPJtO2CusvyWQWuSnobajpe7y91YGzABlrTpHZZEdOzhE0gCD8yvY1oEYJwEscL2dI79ZkCSvDrIt/C/v5BAmEX9/oCsEMfswm60TDC8CxjqZxXdbL7LmOpqKc7ZxhDj19xnpFJkWVCSj4TD21bNMZGom6ZxHCuKhaiqpJYUBxCVvYXtdbuPOw6t0X63pnWzFENeGdsgZddB6Sp0G079gvdjaOvnEYRpe8J3i7QP12i2KuvHKcGzQTMiwMbpvx6qG0/OEJImMTpNNuhTmpeoTbI66SQa02eoHoalbWEuJoMT0fvNDlEiCrjmzxIysuGIQLc0mmSH1xz9gfUQFAVu6w4rxsBaXKxmIrqiXNbGc6VEw81CN+Sk+pgo1VcwGeRvN5jOUXZrWmLH547JlXBY4ix/vYnMPoxISNeivh/+/9DJ9aH1//4T2OuvRq1ZY4YZTBEQs5hGh7+w+C0SPmCs2YOQW4phM+3flXzZxA+xQ5oY1ULGgY9ZgKjH2tRTXh+ABpmlRy9m02qujiM1nrRkEle7+5AYsqO4dYnFT6B/QQc740Uj7vtNePZhPf8Bx31qhUXo2Pp73gtOHrVh695w5bAPQi/rw+IoZdrK/jXzOwKwnujWzOFmy7fvnV80gqm4AWYglcIG1hg2IDrnQW+Eezk8Zp1KWmwWfcT8SgeiqHxZcM8zNAKVQvYd84dJFnCDnZxl96O73ZLLgazw8FXDRNeX8yKfCN5ve07vezcN8zMsMLCpRjTIcmorj18dacNYEF/phnRggSd0kJ2Xzl8JNojuXQHA3svKLtYLaqygICUTSPSETYzdC9GjayL7HwbZkQag4phSBtixGAeR4bK9Gsy2WRrjmqjvN4PPN7trmfNeNm3sHVaqqKCWp/MMA3PJOjnAX7NoqrfnmY1ay7h40fvCU9pPf8fPLnnipV8poxe+rhJbum6YGUiIvFkJf2PR2C6tRIqYLIXMbgowUWX/AixkrhqbPaVRNyym1Rc8+uIYBD25Kf3SOveXO7/+M/PLC1SxWBJ+lDdrASXHYQf+tgDrcCi6DWo6jvj84JFRQUb2wEto5fQTpqzLCur4i8obHo4wXu9W8MJtzwQeYRhMovtGDOi7xlyZREdiBoFBc9FrQVvoimzdxpEDRZV4Nfoha02uuQOVDxgRYnBvvkLhivCNC0O36EYV9CSYTksPlAhRqC51tPLVpDR53H8g5gOxJDi3+MGglbkt29sIUa1+HVT1AK/+hTyieRztbvBDYDB9zHN+q3gcRgoiwOM9XGo//946LqjHuyvWy2+CwyhbA66rbopU6HvHBNjWZMnrxuXvayf1uALZFFxMPwb/bTsdzEYokmVCeRH0jDmzXSa8fwDOec7H79gTQmnGDvM5cT4FYSPRiwUvytmScTTz+HY9XBMneuiOOpVlNkulGqlGLNOXPukHaXxszft5aT7EYRHq7QuGoiKZ+giWGnQZoOBYa+K4ODq2x3rV4Z6SxHGENuUyHraBpx7mUH4ocG0NOvDeSJLmAHwrdGRIwivdzbo3+2SSizBcg9+zeu9eIMuO+GRkZ2gbHTkh4CLGWbudyjx9G3QsKfvpMVsu5wOezwVeABL4ut9Lbrd3pR5SrlHxWu0pWBIVgcB0TFnBlMZwlHxFK9huHETJfnM718F5fvWUHw52bnvDbGVdgSK0H1LBv5Wgf5ofyNBaScDiKoLgAQhfFibHYgIGFRAidIRUHc6FRzMa+25QBOC8P6o2s/LMj1urf6WkIrRIMfCDb+AeA3hvNcHmVE53JDHsXvQICZZ0NmkEvwgLg7z/tHFzkCDpz6POCaeLZeMJEm61+5Mi3YQlycTo0yFSwUj8sFirX8Sga/nfZ03COaNO24umTVfYUvddCWAThN0LU1kDQaU3XaigwnmoR3e2qB/u0t2pt9BeNeNgoUuJPyBRyE4ba07Z6RU5LMajoSKrp3bsSjrOSGqWD/u+hEnDg55NQNggbKQ626ShPrjfPJ650iSYQmeHUT2PNZ7qR1smZWp4yZ90uUOAGZK2ssoaKlmCihEAcm6utClgpunprpKzuj0S5W3CLrDTnxGOaxmdHBKYT3IcSGmkOPzxvkMtOqL7aGsAf8AK0oyMeOUm0ewWOM9faCSnho0C6gUqEKeNy3NrJzbPjCV5qp8WeTc3vTZnYl1zcdC++DVnuMMPAS9GKdaA130fGm1YcO+ZB7sKOpDx0TNMn6k/oZ86fgtMO+dwdce1UfvkL/Hl+mV1B+oBHe2vbf3y+e3+0cTWjnhPJLx6j19op3u7pw3j0vJj22whF746Uh00+ixF+cRtTHYblZ8qyo9sSW0qcZDDcKLlpYPBj/LTG6Y1dvuP/XY0eMiaiiM7mSE68CDwP21N4ZXgs8Ts6M2TTmEGundnX0qOvFQIbOWzkkqlJXMsk99J6WZQKynWKi/cDGWoraSB7u/d80Ev5nIBRqPjZCZHg3DP+sCNJx3GJ6U3sYo0Lcx9nPIh/oH7gW3eecwgsi4RyoIkc7kapWCBeRlx/D7S7tg8iVRJlqk71rQDzzmA64FW+DR0NGoMrNvZuXukNwIMsjnw21MtaoUOU/rRMjOmqQthIPxLdbB6IPWeaNcAnAwPK8dBnWRFoeiumaknuRNNhGVTj55F3azg1TcJnUS4UswOsz6HYTv1gV+j5dURkElKMmXp6lKVoVZtODpOfo1T2ZzCMW9pH7GGEyAnSqJSVLjufHncl79XM7TdXWIozgAiNKgCMT3FX2a5DwNrLOdCqhRflDjVxB+2EItwOPNhHMatMhBeLOLGcKcCDIG0XEyCHcHT+gnA5wgvO11D0l4wXgQe+zlp7zJozkOiiYthCu9jaeFqzZkhZoPH3bmEsOn3sI9Yag6cg75IghX8r34sZ7Au2QklgThZJW+KLwWF5yqODom61av7xOvWBD+zytdIscPgOlI3pWOq26jgSPrdRUPm8Qir5P62Dz6bw5EQpWFvACHlnIX/qzJjdFFPOF61ELoXG1PmjGTx3k9Z/Bd3+zOn26AVMUPnQgpmhN4LjhvxtyWIsccEDjlyaKpIuF2djnjTKdQ6kPTJaBlFgnuhlRPRXgLvspf2kvfBFPpNUrCIA/jH9z8WZXbc8U43Spj+lFtH3lPjF9tupBOtibvhnUo1kdYPA6oyH8K/L1kr8Jwew4JHHMXTRLmQj0yj6Q6IcdzdDVxDulmigM0iVqH4ygVU09jNews42Xnzcp42d4yuhJIm3ShCR5qCTwodgEu3ZEJ72zHEpsfTyr4GuIj+0CYswPU63NxGITP3qRJEH7qEzdBNQe5LUaA52R8IES56eXia33oO7lW/NAJAV8u5ktS8p2DsbvRePPFaCLUYt2NBwLzFK/0Kbqj44g+7GITMl4xkWKeyKtXyvXxdMtMC9RfFIwCAnNryZJ6vtERuxhnYNzpn9BxwYDnQugiN4bSV+DRXzAj5WGVswjwgT30Bdj3aRvqEDxLiwlPjRjncLmjcYu8jddBsucqgExJ84i7gENzrY6keA7DdKoOrwEQLjY6qJBjXsEGD+kxSSbNkyxENXXuRQChKMrrVrK3gtlgOW70Gu6rC6NSiTQWYiqmuNkZNga6zi2PC5LWARzThoeNC62vHa332M4O14IoUlTS7vROwHDO1UdrsC9v9iRU+Gnpizvt2tmuVDjhU2CXKROENz1SFOAKJ/Fb/fM23RL1cdNlY+ImxmzxmSBjAp/It9EZdN0HbyoyniazfAOnvpmIUgkJ56czcUGqy4U4hRSCo7bJBwCyVUhBOsIjFWaOqQk6ErNHC8L7Hrk+aSWPJqyaOq9S/RmE7hmf/r/iUHG+tPhzSjWei+gAMzLUxhVP636KR3eELo/rOXSpQtWCEVFU1+h0Mo4LwOsiS6K+RcLIDPvs2ZBFwk+/5TVPtNreM2sxAXh0iR/9QALZv3vNFY99R9Vo3h5VTXAoOAIcQhDI3BH8zGPKaH0NfktHyw7CF+Oth0yXOhxSzURUOZoJ/XT8xGC4gil6sUdeN+A4aL0XOcxoD6ocLrc78UN/4iv4qAuITsAEnCB8D8Woszv67/ZUaBNg6gMhqo80L6qPwC3fhnKZJgs7mPCPPSHyTWGivicekTJ8KHtQ0GIqjkqn0x9O1ekPX0HASsrkcTYpUqfL35+qy3/50+//+NWX7It//PyHn2SIX/3wFdxuRZq/0+VF9aIxOgFXrX/qUpXM3u2UVug9I12hVkEtIsq4E8o50FV/oMu7em4mUClLe3qMtnbhTaZ6ZjYAh0iBZz0GaDUgo0YB/4L4Pc8ajogTaiV69erhEkwngDi1cwvvoDyqORp4Ev43AucfDMGCnXQ/gvCJJZXz9PhHgasZ2BKt30H43CeLmm5OwI08PeTHklFGB2B3A4ITujYRqu1dxhqIzDjHuoANFpXrLlJnXR3D6RNiOaYpU3nrKXiSzjJGKNpHe4Bpf8QnyWIbXgIEsAHuoZQEakKboRR44poIWGdjWb9vNhqyfH4ec5/l88wh+IHzJnvnMJKpyL94g6DjDixGrTqpwJn+ygXRHAYtQ8guB1xzy0LrK6p6z/EKfTRunrQO1TDd7psdJnldFdCr0+lgqImjWKCzkecx2i3llyvDnvtuWLHZyhg+y2KL4XudCSPSPQXmDex/OlUuNoBdf7o15an0MIEsnm7VVQPMc63tFc9J5ztENdLHDYS11q/L0qI44BC8CK0pqxjXqCvtT8CAB4NrHomzdXVMq8ZbgP9CuCFH11RyY54xx3h3pePMRK3u14QcnIg1X6bR9ou3ZDKt7w/lPzLUX+YFaGlu1Dkc3AnozpevhCZQnVK+Ydl4ld4p+ULccBiQI1UnGSVT2lE/BAxJ083tEQKYVfZy6JqDi9JjIy7KILxrx8rDq8ybEsz3cIBGO+2mmeYJXjG0aOJr23btxIXOGPczgvDeoM05b4HvgmEbMxrL21zSV35jM6jvZVXMYItmCh2gTwzCh97mCi+jOxDoh2kjJfeJQXihShHEBgKJce5fhYByTBhUqQs4r5+anrBltu01Kw+hfr3DnkNaokveJvJVi4xwFn3qC/ath/rxAHXbS/X1u+ftYc/bw563h11vDztEtW85X0zLvT45O1LkS/YGCx/OFSuAUnkaxnL7/jBPpkuC634QKVr4eRqEd0bkvkOQjrOkQgLqOP01F0e/oL9AczrfFrqKRSpq8du/wq8Bq4EKdy5X6QY5kh+xuMi6mCNawfuejTHpIPxmVR8K7StDoPSPg/CfThlcZ4Yhqxf9wAIBaSCl48QlBeEjiwS4tK4Y0IJwA90krTOADoiwijxy/SfSkyzeAoy4fpYkvtqjg137sgMnD2aeO23KqT4T2gQNcdIZp82fQUieHjqyzlowdLLm94IOKQJJ3SbBPPVkfMGKkBaf5AmkX2a7HfQGU27Xd7IJ5OqczSYSk3bk2YyzQwSoAecSeJbiqptgqPjfHM6zl9svDJ9Um2gv6zgpHAaFPwKjczzFgK6lsJmhQyNgEvsB/4u+2AsbCEalGAECjMryVkWctHtGJb7s9OIs64LxEoxsSQbTy0Brrwt0Fz8Zd3ehQ1UfwyD2cvt5aDuqZJl1TizQqeAlThKwpt0bcXdp8oPOtdXufi4pCH/tplQhjlfeJmydjPIx4G88qbmlPl4m2WUX32xTnxU4o2RZciTUrL+3AszfAyvEtOR56+PpSK2PT1rOhzUn0BQDUYWOdtVk8L5hwwnP5brNi+Y8yaHhXR9duS3VG7hpi7SGcLrvyzqvTEwFfO9YWkDRRDXVi0LrxRPGa7MoQfiwJ6NSRCGrAMIV4P8gtEqTHSHOiccl2PM77maFT+6+K7fjEdrz2RIodhfN6bpISWViPYaWR5Pg6hgOwo3JDSzJNPsYQZ8847DQkE0nKT72TZPbqsEpn4jUugI5US0rIWZ9wXdzzRI0E1xMe4pZL6SC0GnwLd0eEYCPaM3ig1Mj5cei0qmIClDL+BWEr5b5Zsf5Dyz2pJlKX0CwEXFMYc8Y1F1IWMzWfTwex262YiYH0AtujwRDw3K7PMYYoi4gxnh7VNKonNFV3lm1RfvnQAvTf+sg+D3wtFD+TWwIhyIR3x3z9+bsYGG9BQ2Sjocl6Ol2j2cPosfH5waTiyWxbItP2vyDGy4V49dV4uqdfhppW6yRskg3bc8yBX3Skq8cudO2PM0V01OsfcRWgHf3990Bn7ERQH9nVAQ1mSEJjKO/P8DFiYvwJjxNhy6iyt3wVFeg7F6msW2vm25qOAJLqhW45dBtf7Y4qtUHYUMCnfSJ8AJggzOy37oY9k0jtL3VWol1BVnRzqFJXHeD2TEEXxzfHIASo/qZty0uhaVDGgTg65dVh71LfLS97O8DEu2CpyKvb/jYHOz7UC/ViZ0HI3Ae31HUSiDgcc56UIeXndj6UoiD0KXhsbCskgwCyDuPQ1cxx5ZMppZQ4BXqJNqE16uOIKWOX9RR/ZB6h09qTVEAu5O1WJ5tIgKYQmFR76N84iHvXpLN0KgOdqqUqokYVTowTO6Xbey+feCy4/pdXhA+anm4FTvymhaEXR0CBUgG2pMjbXOC8NOWQ+tOWyHAadfnBuE6+irQZIfYFrBusFInH+hQjWZ3xyYkLmG6u3OlJWhYmONSrLlEKvhylnIbKiEIrFylC0xTMEGj1t1Igcg44If95YAMWgwHeUG4YfLA2ACdZtnitQNUvP2CokIWRs0PosjDBEJKsHTTpS4cBZYh0F7v6QoQcPDD82lZJxlvSfAELrZVRzQct5G1sR0T3E5OkeaILUBRH3Q1NMlLBWxy2ZNgTzTjvsVRfdFO4yhkfbVN4TDr3HXUybQLh7HzOuLDq/p3xGuqKgIa4g1FzUr4enTYjKrutOFlxk1pB4sYIN8EPeQEzujjUFH6cv5lL+f/9gAygDI5y4cevv6Fk59evx1NA55j0DDxpdhXoPJEKm8Q79VOViGlO01tRCXojFRHuzejlFEyy2F1em7xKaomqo+0Ct+OGV4Lgr+pWtcp+k5UKOnb08MkFm9PK3FUnlMVk5XN7Z0phkNfFRna0Mm+ztSKSPE2hqrADki7uWUwUAHRAFJoS5HrLls1u0J0AkgrVWL8ZSSCvtiif8prpCdopBND/IYRAqSDulqF805cNJNU9EPPiP4KgFT7gT0VhAdU/4MjOZyJRZQifnt71CuPGTjkLmnCTlsK+67Ha97hL+F7X/eIwPql8Rc84UwU7g8BTctinupZ/XB5IBPgrCwVm2Cw7BiKLhi85cc9PGuzkJKfFYS6qHK/rMVYyYsOpMLXrj2P3bCF1EEYIXq6kqQKAAMDmV1SJwVPYqszRb/syGClvmB9TlLUdilwvOcbFqkLyjoQULbQYuIi+sghtT8cN/uOwXpof9ftg3DDL5QU6jM6xp1m5wn8F4Q3fRLtS/sPxdVHsbpKoCqqN3psVq0YPaYEN/RPzEZU3xl8dTr+Tbv41EutC0rO1mF7djVED9HKpesHmKFFFQBS7njkYDUophr47qop0c7tbZeaIUKoW+arZQThr5ysva54uIuYYjM7fBeNGm380p+jeYTTUZZDrCB8G9MIL2DUmWym0+QIV7M1X0IgOybwFNtIZ+44oNbA7n/HX/0qFm2gw02/RJRi9d9rNtcIkDCQr9kxlxxsiWiGAOgljXQgn6+QukjNlBK2s1KuYyMhwg/LVhWH49fYOeSLtgmhxXw6iihzMsINQqPeBgOfTFRmiz32QhkXITS142sMFYxlV4HBPPOUl8MIEJW4KVSMtl7jz4EJUyVy8Fz++VThdpMKdLR6OOTuX07XrZbxhC7+4dRxgcOxi9+fMgyQoq5lM3FDrH74qzrMC0+Xn60aX0hBZk7rgTtcGp34pne4YofeO/xv2OVP453+7DOH9XRpjUoEX/54e6dJXz0YCGa0o7OWSiFG9I0BKQQRe74ke7pHegQNdNhak2PFkC7E8zHGeMJVzzK2yDBH/zJjiyllq7QZgO8xOG5TOsv7EL7X5FE9jac7ZyEzGw897zFWp6pKz1n8G/56z4iUhGLxzaSe7lLpwilGLaVn4UeZR+WxEUZZHSJALvW22aPX1XEVw19GjfQDccwo2OdcR/xLkeQv7UDLXoylRr4BhTyJiMMnUPkulvUvV2w8heRXI4jzTaI724vdhUYIM1Khza2zNEIIDGqE52EMBZlQ4Pnd/tevv//Tb/Aw/g0qb58Dju73FO/1HS9vQEjov8Bp9ff5H5pJltS1iP+1qA6+LHKBzC8xQY+qbf5hDvPtuyJuUrHmMukqFxF2DP9UhQ0NypcUm3GhoxAu2SUgfE6qhmq1aZC+yafFbyshvgNvazWc0T/n8RuAqMNdv4H4N2ppX1EckNc84v+hTGgqLUwrqhQ8CwlwqjU+nPbnd7z8roiFkRgHAdDOz/CzFhwetKrvVYjZl4ISgRBCa5AXhN9140bEuK/yujp+3K0MHbRUzLp4YKqN28K1tEvvttObNQoPNQjvdMHJ6pXv758Yv4Lws7Fnt79/MsYOwrOHkcSiR/IXh5HM6c97agNtsyXSlOV6V31M4UV3bBm4Ywc0/8+2hIXv3N+h8ZHlECS5EFXKyxKBCHDJtS/87oKqU19RMYrq/ihjdgDKDxJasnEov1ZkoAsLy87fRSvS76J/pOt1YYsMwl8YOUV++AslMDgGo4OhMSzrwoDgGOpCiXi66Gku/S4spXG360JSoqdVu4H8W06E/K+7NqQXKeBDZ4Z2Vx++ZsZLR9Nwr5lR3oCRGICB/seyFtn+Pvue1/Mv/vxHUkTwiugfM9SJ7rpDcOj9zwXh0G3QlZeezISXnePOTUmwOZ+acfpDy1olqKqxs6oNpDMw1l51NJ2h33ywPoev+WMrG+KQlwA+kvESVGAnheG2mzihu1J5Gb3Eio+dDu77FNvMXKxevdJxUF1EFeT5jKDMmBFL9XEJMJXkSbPKJEFwPmtKd0A9EQgUcWspmWVkn27h08V/DJPX4/Ch2RWgVagEdKez74duZAjf3DpF99Dvg/CJed1pUTGoL8OkmBkXZ7F4jacLCzUH7e6ArpYXEPw3c77TbRT2eooo2ietuxZ4b+EKLdAX/OrLRVLVDU+3aohO3dJnBfpoC/JU9Ja6E4sdhKOdJOOdJLqTs4tpQseOf/SgsVKZYZWe1K5AkIDSlKRZQ8yHswp96IV1lXXFqjZxkrQsXHZMWIJ2/jSZAn4jHiXeVDx13tGu1di0NiC0EZvAAcFpc0XjHyGMUF0gNtJFAPQDHJOqKAjT+bJR2xSKLAHtkUGjMEdVrRueEKQKg3ez4vK6IefW61kzcnqkwJJIIHfRIFcibiJxyaJEoNc8NEiQ4gMFkgpIo0vQpRs3mFYibxtinZ8KE2xYKvj0jjetCFAOiikZo7ZMCQFBtxoTiomjRNbSHC7FA12oJXmJoR4MuO0uagJEmCClFUF34nEtPtAELBkLTybl+ayBNSAnLRMOljqGlsJObzhgVNbPu5769sp1o824m1bNN6vk25rFam0U92s+m4nYSiYncxV4tPCNR/VZzPWHeaozqNQZlerEqQyqEj+9PQV8BcC4s2SKhQyvEinj8S4oMVhbKYlvE/UvJTp5adbiQRSycFgcrQ/wN1XCVjJt3T5pUWHqnGIp7bcE+EVcdCt+eFmBb5nwydcVDaKisOqdbEFU5a9kUVY8L7wpOCfDzCB8PNZS3QiSIblqqei6kzWmUc9uOnSMl6FKt2Uhb7hc8+dVh0kq25pDVVbyL4yyxnryRQC2oIKgu0ynIYkg/GRpHyranw45PME43DoIfz3UENO8Ri5NmGnhiyXtBy5712iG70HVpoZoUlxpgvCRkVmHH6NsP0blPE05yF2xMvDU9WyigoG64WbfEZdmwU07C+/E/BmEl2wuBJ3orEAb2E0izPOaxWvf9C1FJg1OOWWyIuZpAjh9On3RLfLYLiUa+a2fuUeO/vvebDrM2gMvBL0eP3wcOp9xXPe8fLv60C2vjMryCcI7bvVJp8hPh4rnlIxTGCWX1M6mAschBgRK1efiqLyEWVVo4KyO2WtYHlUV+3RDRSVhyl1dqErSbNIxIOsOGPD/JybjY8X42GXsKcZen/FcMZxrbCvGtmJcVHs85GwuEGz1fMVzyNnSwCWfjuYinoxwg7CXbOhL9NvxJvqpdMV1X5GlSkyvOfRddnQk8UWfbzkgfSTPGb93mfVzh8nLVsYfaFI8/W2X9kSa+/6+kQhFiXcVGN7GTGsPw/eUwRsCcY0Ewe8N/X9//8T8GYQPRhIEvwK4gARiDN4tixLQWT8bx0A8GWND0uJo6xLs8RB3/VaZ1z+qHERQiM/Rn2Ach/iIq112ojJHw5/rZqVjpUCDfvq7vzJR/aTjBeHvf5Ksd4JgXIj0eLS+uO5vKOFRpza+XLkTXaXA+PHqzUsOmL9WSqv0p4NS8UxVczoIf3P6nqgWeBCu9KqHB0Oqr16Fvv9pXHIfvsKQo1zWo2XLlneI4TWfrtJHXBdc71JVIhsoHNnk8X+droLbKbNdffCjo6Mf6Qu+mh9P1daO5HEf6WAkz+mG7fvYR7+P0Vp1mBYmWZMjNp2IV30Adk8/yQN4YsW3U/ynm4JM1CBct6h4isOv65ZFn1DOVt4CUdxxc5dPbEIQ/slNQrZvreMuuTVb8I5hJ1PRQCpgq4B82bKQv1qhUiDGk2B5wETZc8FucMtMkO42PFkLCEzJbw+wdYbSpp8PtuD7flaLPskmCVfJ0ErIypu25a7YKdoElXLZIcJVr5u52jpsTR1erxj6CgSe4Rt6MVZEZpjXz/Q2miEURRB+1s8GXxRgOkhZ5a1QY7CDcHcwl9zTEDk6cd3XBp7NWp8L5M0+GcPyikP59VCdHKhKV1RZ9xGMyQThJSuXHePCf5FN0Hd4GMl3skmVDiSuAwSjlwH24XUo9VJC9IYNo/wA6EkOWXqVaDENyYAsWqmHo/nwDLIe4zLK7i4RSzJ+pROBYyBBWBntVLA6PhOojAVhemDl3zREqHZQhUcMXtVO1j08Bisd34CIhuTbJhUPuqR7+Hbgf4y+1sOlazto0dgvRf5fsaocJVkc1bsvbhhp971U/o99Ofnd9dgkwfOeCr7Qj33D1wosg3d7hUfBSovqV6xwINf7Ikleb7/Y6tFlM6FREKhAv2+O0PoC7R9J3PA0MEWUPYrmDkEFtIIX2wI1lXJ7bRoUZ45cMViY9kJLo0ImGKsKdacvhOV7BC8h+7NosJt+fSlH6COvEFWVJOAtJFKEPa/nIcjzFAO8hTlzFC6D+qCSfAZA4JPkdYNpYogTg7GtiCScg9kR4KDlOSSC2ZrDXx/6SySc+MiYDO+ig5+4pCC80ZdqNyi568VPgIrnfmQFyNa9O9im7ff6YH0uTM4fq90ldUr53UE5AEEswdP1cFQEDjmYAjxcLQzEMgFYC3dGZSABZXw8EeD6Dd/anMc8A7cXPp7hceOjhP5UWnTGplPIjKNwLbUAwulm3UKe67SwTYtuIcDb6PAGAnzyo/AB2bXo8LD8BB4BCM/AfDcT+eCmRzAW+joKYqDJqSY1qw4PFjR2hbStUpNge8X0tt2dJx5YegcuQ1Hbi+tNDfe9Tk265uEmOQTj3PZwdEYB3IKPjxAb6sqX+nynGDOBawDjnh9SA5I8W4DdO36ZsiomgibOdb/ERMyS/KmXp2eDQ95oMTwg8RA8zyoZ/hwxcpUU32L0twAdqgMLXgJ2Z1zOl6FtwIv/pSmksu17Qi88QhYKSFdXQIWQnxCuRvh8VMoFdWG8DtwGuwNQHD3ckN03Fhq6k92PPV0tkVJ9WQUAy3my2wcbsQotYDKkgnV/YDPStMk847DKMWC98iV4JhlP8qHpYEGk6NCPkz4xCNc7LFYz5iK0nvD8mC18r+GJJdTkxw0HkJtcerBYdkxZjdpC5RaceaNSNKb5Jys3AaSmvCZUlmNRPbMaiizjHciP5zYumOLgtbCesw4J2fNMi2cDgtbgOkD/7VXEocpBez9frtJCBSEauCAAJKE1+lp8tEonc8jwzw/mRZU/XlG+rMSTFUVBEXjkld3urS3ey+8YeS3dpLK6JEApfEc5r8G3r0qIispE8HFS4NZ76Do0K/S7ciBw3IIhLcMoieK2oGLpaVF/YUn4Cpd4q5EYEt3KO9yHAeICWfWfLW1ApVUgsEJKxCHEEAwgivjlG7ae8XreNn51usbgyZjxWmwvbY6YKUYXuwMthh/scDUZPTg6zDBdgusTn7gfN8lmPvHw/ONq1UlHFtT3Ktsa5IE3PEvyJQL8KPAKeIj3vYIKM5lJwtq8ZQkRaIDhhO4hPSkEqGdjSE/dFAZp+HTcCjg9YKicji3DZW7bJ+fh2U9uSKB9co6Ah/iFD/DJqutzskSirVs0ADPlbd+2feKFj/K16dcVJlnjsbjoVYNjaC9/x2b0JHsXNTGWYB+aFn0BA5sqiY+kF5yqrIQ688ibPv4B2tz5TPRG2B43NfiVkkgKZUODMjP7+/Q3bRGbwxIfOSwK+G+fg0PHY7RX3rzcRiuDT4muBn9+aDP61zLILfKhgcA1e803HOLBQmFXarwudcZue9WEFl1QH+mcmLUUTW+10PqtK4a+AUy/ZIk+7vRlMOFNdILrtmBbjti5SCxqiAtJxbSm/RhsTWqFmLUFm+nsZ1ae6vPIDpXU8xt9ia6i1d1BZtt+qy/S4c+DwK1OwKiKBYelHNibw2yFxka4ZzRxKn54waSCJnXdJJAq1Arf7/PIZKlAVNBvca0vRIhJpRp7t4y3xZ4qAHO44bDVlkE7q8tsrWrQ1s9UYc8PXKb33Vx1pKhKi0u1yn/bIGsq/op6vOJKwBO46hIRZG/DperCNPccRvtttZR2mTRkcHHTVVFbHERDQLa+erjIZo/f4gGHJosi7Zwh6PA7axS6Xpoq9QcTQGz4BNAp5mPgY2lxPDU8QbQQUQ+RDohB+HGHONeLqRliBeHlgwXCtuzvn6i/gvD9gyRL2MEuk0ndvIsx5q9CG41OhyBqr8ElF8OO13aN9rzIkd7kyesGA5DnQegFoQNJcNB4Aey4ZBiuHzjj6diE0HJeceuCcPOuOIh2sq6C8LpDxNrxuLl13WteIlkDCBURZjd6uBjdTtz7LnxeSjGjEop76wtsanw6AO9HkGn1V79EPe0SVHHdxMJ76JWrAabfFHvuFcPEmzJNIvKpL+83w6RUQ3DHKwafPuS75GJG6ddmzx+ONqHY7WWAgOBjNMQee4V4HCe9+7roiCZSz4Bpk0dY410pQICmqJ/CFIGqWvS/E+t3EK63QIFmFUme6ukf8VIqGL1KcFnkmJMwnRZVfb8voqEDO9mrhhCG4iIUj0ZapFbTBDSIM1joLPlRnAHlOi3n/ALMy2m20wL4vQsEwcszifqOzievdyEEES3REIdIv1td4QLEG6pAEgZP6aL+4tuSeR+1SSJxIqNK1FR6tUomDbx/rONo2uefrlAXEhOBMD1oXBhznrDFqsJtzw/HhSkV6eGryyBmYB3B3XyY5DqnhX1Ti+r3pZuWRnCTIPz41fv4cDNSGi/gjy5uVa53AJLgwAT0yCRNi8PzYE5EMD5KPRkEkXw9AiL52gMi+doDIvk6CLdbWqsxucIWw4SdhPmRCpWebzWxOUF4CzmAGeEmYBIsVMumnCzdFa+qILzYQ4I4hxT034AJ2cGfRIRGgwAbw/st5UAckzz6cKmDuyb2o8YJTKhijDIfbpsiElECWJpMHEzJjhGEewNgk3oLRkOO68MKBxoRbOlCRHta27zcwlIq3Mnp7s61loYlFYopJIkcAWej5cwECL9ukuigY1BUsa4hjAvAFZfRoUqqC0KCZini8yat4AkhGLYXB4UKTKdS1WsVtN6cI6RLcMKBq+V9BXwp57wU1CMMhPbwq+1vpdk2si4ywuWE0m2igpoo8XVNgBU5mTVFI1vwo4s2QmYuDq8ZAJk6Q5Du9lKfs9EWjYVb0iCGyY9iXTGyhYqQP/oE/zDpGCBP9G/79L0e/eMB+Y8H5PcG5PcG5Z9r+ecu3Td+ou/16DsD190e6Gfb6Gejo1OFGz2gHkO3uNYxVFy9btLn6DYXFaeDU72iqv1CVcC22MJNqwQwboGYScawCteGxTVK8OrqwASSCrCoNz0Aq90HQCV74+Oc1Ua14fp6j4wjANe3tNFZoyKb9NFZ4cImtUNnfV+jsdZsO9725sI6YwVoV9DX8SR714/MmvAZpZ5Nd3fOKZF0J8nYdkyfOFg/p410qv2qNQ0LBmLizz0fW1ebUiecpz4ZVVBdLBCeuau0bGOx6o9U1Gwn9rJmyPLWZ+46fTSAJgtpT4ixqgLf7g/IIRaGFtr0CCm9b83DysWhD4bWXrMQDPfuAJqtrqEM4HIeEcI2wbiQEbjbRCHeRk1lY9YC2hikWGz0qKqs9B/7ELh0Iu7gfFX9nP39lUXDvqS6TVSxlDi03bIkPY/NfmkUu0s1SIyi1vY9w50BtV8r24Ag7jfBMCxf5e3tFcpq20i/Ya8FmrtwwTIaSVC6vJjBsIXBjHjQZ5vVuMFnkPJ8a5nUhiuAVJ4JbznvmS7nfWOIC4O/5kEzJpF3qBLiFUIfkAcScQZo436PwM4hEVteR4TtCldfV+7qbJJPVa1ERLnE4V5CqooOSkU+q+cBmUC2AhuGSMcCdH+8PYWQtrenWRGn26ir0SkBABh5ah+LtIbbKX23qUUPjZgkg/AiWeMYhVDD0nxZZAAmRLBUCkLokg3NDJuIi9ac4/8ONRb4/xWbSjnJLhGq/aVrIqMi6qAeA8Z0jFGG1w0w51jWOmQZM30Ip1mhOysZjEWQV9tKD1TIUEyaWRBCnfb6Rynq8zEFY4BKXfHq+BcxqWHyOIM/MQspCNfiZDol4G44jOq/31PRohCuermrTAPdYUj+29FxlIqzsArC602/N04tZCMkq+apMaI0uDHGy6Jiiv0LjViq0Q8iLOGBYt8qVl0lAuZPLLyAsnWVLElVeNVBs1JfiI7MeK0hZE3EZVxl6Cx0w8OWGq1508NUEbIbHhZgARx962HAs1IAw9oFtoJUEH43JEW1mGnHGOnMFAvCr8fFsrpc2lVWl0tHJfgsFbtLuyKxpb3FqLIt7Y3EgtAHwU0fEISoo/vG98ZbfO77A0xt7wcUXumrjo27qVEhuwX6FrWFuatBlv01hzXydGdBmzQxArNiNHNMYBMt6nYfB/xkiGWgZw9CjNezWsMSE6S6cpLpVMiTYWYQPjKYLlZ4k9dJipp2MQ1H5Mrjeq4tUEEYLJckNPMVBMlafntMUIiDsY6oZKZKeFwqSCf58adCqWmA5RKEj3ty3Y7pJOneMUQ5nB0qkXMVVzQ5ZjWfPXdw0lWsa4sV7dCN27Ew0z3EazZYehG3+AhX+xwex5pKzj1dGWnNQ4UyYH4c88+u+3DJcTjdZbFwAakfyY+iBenWkIEY8aH7oYJrYOPBW4N8cfm0x4NDjI0l3pI3LaoyOZMx+ZcmSwGIa/fIySAPwJv7PMo10cDjt70SMEbi3/Dw8RyDj3Mc/3yMe4GUwhaFq11iTWn4vAQUOaOxeGHU09SAUbeQ2xEXvcvM0tjpi0Qm9X1XECM492yhh65Q+43hO4OSUVCSdAUx0Iw/GhRTi0lRsc5S7rwZW74S0+ToziC/KGu0uVizRi0Tba3OQZ7zztV3SqH402xny2Rmcma/5kzOrEeLx2JbBEl68Aho28HvG1uf3g09oPPtN3zNAxp/yKusKW97OFTMgsJx73r4uNc2WXZMruT7HpEeyRiBBlpXS8i6ATTffEIuc7R1ugD0+DU1n/zBpjv6astbVgDBEiRQ/TbgRiVBqqyvUsP+BOGzFTDq46Jd2cfFCZ/+RP8JH+VycZnzA7H/Zrj3gFGWy7KQYju+78W4x7YIq4POF1Oodem3EeHTDC14NwwhWhTaIPksu2ZC3Ru4+rquABZrxleIFm6MEbuMHJFHRh7vOVSTwOu+vbe788kZ/fMfT4UxX4sqcxDr/nSqjlQ1YgTz4FEtKqfXfz5Vr3mRIwL+D1853f3uVN2JCELte/D8fzxVZ5CuywHTyI/Pf8pO9eNTdhCn03//KzslwBVv16d7QfzwwN/d6aYjrwt3On53uo4kQJT1azv8/nS9qTDBZOHe5ven6u/3P3zJvvjHz3+AEg9Oh1uIiW9hRSuU/CaXrgDhPDs93HIEXJxeYq8CRVIXOHWkrsr1CzI/gclprZeziGffqy4Zqdd61FgSxsmtriIDRh45Y93r2JDDrl4poskyyC4iY44TATBbuSQDHaD4EQP8qgGcHgNsx+GVp7tOm6j/xhf8aLQcxIE4VskQ4Czuaj2MyUNIxXivEBHQaRPLe1VFKV5dRSkjegN+L68+gS/wzGRKSJgfgIr3Yk+liW+bdSdQPU4gWZ2nxzKRGHFok4LwV70WyuTstnLIQfjrXktezRosT9pr3OcE4c6b1sgIwl2zCfb0o8ADntSWQ9C6MASLUDQh6XikDX1ziqnM6DqDztesNfC0UvCywiejsvZlQp8snnjz2pa8yERGuPwiZ8eJSOPbJkVhrGU8SSfFEao9a7golMwIDQDnyHtYM4SKdHRVQeR0Z4MxrJXLJlz7L+Q8yYLwPUTfjeZV3kBNkQps6UmxM73Q/c1SQFX+RUfolwWh2h8XOjqZ668iwu0CE5M7gLjzDHzikPEFhxCWnmEsa/KMlx+oACjUM6t3GSiB0533mcJ4gdCoi4xNAYuiUDcpYqAgBrRFKZv6MGop7zEKWAF4QeDOhMONMRBEVNUZxuJkUU93P7XLhaAdDtV8HX5oVA0BB3OEMbO5kLWIf71KqRFAvm1ynk1UZEjZTNIkQuGI/fu///POr7zgAnlBtudftQkWX33/7bZK16T85K/++dvd++x5lH5Tf/XVl/z7P7KjZHuPJOQhx0FMfrJrTIev8QBKoUAVD9gnqRjKiUsKwmt9qX/lSf15fnz/66//9NvvRM3397/+9s/OsqhLcH+PFcbgRDBzXBMdY8lRzxZ8dlhUBxxRxuAaZMzAlb1uqryzDaiqXqEprowyZImqhCyLXAqpTj/vH6rnA7gtH6gfVOj0PfVLHCV1cOhEuavluHUY6X5vu4IIyKcxY4LwoctXuL5d+Bw6IofETOcTlwePXDEeEQ/CI7RntCjl1oAcFS3AGjcDAoA9OsgEGMibA8xKVDw/uDfANUMRrw/IwJ59Z4DXVaweurX2gQ510T3KoVsgVfLBANcufq2nDjHXAO+StbCcrz9hGFJOZFXTmMLygHzXIstmMoOqMPCjgiJhGIxuiUD0RA11tZF3zeIlcdfxVZeD1MtI1eVXiXYbaUagD/ZB6QnA33T53eWvuiykXkQqfHo76rJXTEp8SMR18FCb9ZRU3aM1l04Vki4DWS9LakHaNGk/UGiQ8nZftlldHSYq2KQolzrK16LGKlCXOse5bn6+q9b0J9gXL3W/v+Pl5xApcrMjfS3qL0BF+Q6jH38A9/aGVd0Jh0kXtYMnVPwSBS4BjHwV/9tITSK1rLrVlgaXVVvwj07PP0mnNzwljFqb5qa6vy79pGVtaBZom9pBA3rVYx3TlchS4dXKaH/fR/03H9G5q1Zi2V1Zgv0CPW2dqbbSkFOgxxbol8bpP99eaRxb5Le9LtrtN/J2AQgIuVBHViwh4b+VDkt84FZsgX4H9DevkqFiRbbAUNmmOa/E4LOwRQbGYJVt8o2hFRjoANIY6rEOWgFPBy1Y+FAHlkC/gz58ptOBLdB/jKA6kqlv8DHaIv0xdFm9A2OwBXxjcDvwjMEU6Y+hzc86GBiDLdAfQ5djNTgGW6TfxWBRImc+tIWJvCW8lIoxcBu2gLf+ltOBr/6WITJYRsyoROYvI7a0DtnYTRgCTvmu8VJa6sld79pUAvDsuhpwrzb6PLQOujW7OsOm70KqZpfThhKEYC6OtPmoa9NlAiWQEwlQrDxVqyuej99APm+ysnkzefYm48GKPG8gD6eA4zeRF1LyN5CP4PT5JvLgungjebWqfjZQl03ZWjqaMmDaE+fLN6zqRin+TicvT9cJzevPu8YYdelUxzWadZ+gLWKoCUuqyqnVpK6SRdKzbBsD6aIYVWkpO4SxLUTxsrvnZ3aJPH8PjElZaDxqXaiue9XPhqrcuaUNu8Y2x3ibTB1nKEK4p2+YOqDDXNKJseb0O/HNK6cTF1fa6sRmvhzsBC1puF15OqEQ4XtmY6i2ADijbrE3UwYTobAE0cuOomeOKhSVxynk/lYQBYC5246oddF+ATf8EDasbztB0RlER1nV37oJZM21dg7+DoVXcXWQDyCp55mASsY+HPnPV+6sDRZyetjDHpRtyUFBRJcYmj5edgLQiErjKcVEFXDtLtAFej1raxdiVyqbAbMV8EyP7o27YwIqahxE2sBlZSXGqG21pzVl2q3ZrWCaTKKyZPrgBikozgr0z2OV7mg1Gi93Z8oE4ffLqu8t7a+wO/ztaIfTJX1NVxpXssq4kjcYV7KkzmBSrNbNkttLpl2pQS4p4fmCQm6AlBF0ip7RJbsuU5RqGyULoVuvOqPwvKjepEYgrhnfecsRjq5GouLSXoHahfeLv7I3+JQue/qY2LRTFp6gS+msecdLfrpLuGdI5xL20/mNfQnzEa3wuvBb3xjq4qnNIJedUr6T3OnlmVMY0slysY8RQfipKf7QbxJJPFshvM17OqOCimXSPkrpwrCUgVVK3h2RIfyTnQdjIjtsXlTJj0W+NSLFp1khNocFLntYFzQtFZQysmnkh8gumJfl03iYFV1zskqwtCZhR3g4V00aBfQ5Vwb770yqFMCyvtZnqZDtTV8xT7ryW00eF+d0XU+qSXWllmovUL4wzL3GymrkMFW5eBe68pbU8JJR3TJLpEzy2WWreCXhSUHNy8zqCRpiBR+GPmzIH/igI6XF7H73K5EMIz6jtAGYe8xiIk/SlU6IYNrTJsuNfkQxvdj94vECTcxEgbCHw6KK5Tv0Gx6JkaR0yfq5/YKlokeaYHnSyLqx9bpKMsD3QC/T4TypBYbRPsXHLRNErMQVHl3ckCxIUB7goM0hQTTi+Vpb29P04MiNlkxvSTYTSphYcxhEldecmqAK5bVKxDuEqn3fUxe0R/pzj2JbevtWzQFLry14xS44SlhGtyyi8v512N0fjpQcPT6ci8qqPbohGwBXg8eC1VvbgJv31VzHT/0D9aMSpeD1e+0vHuu//1Ikuf4bMtrOyLqagiEC/sDT7R2VYCcJ5YZlPOczGJwESN1ZBIUTsV6oLrWBtecWbGfdQ54v2I5PfM52Nj1kDCBJIlUSlYB/ASW8kLUumPDJJKkvED+Jo5qq9qhL/NsXky/Qt//D17+B/t6D6GlWibrJxaOxGqNY/oYm2v2lcjORPxgV0t72e2NSKp3j4ZhMlxA12pUqP3NG5oRj+J6c852PX1BysER8RlVsokqKKqmPz0hR/3FeicPNfn1UykIMwmsOq0O8dxvFom32g8WyP62OteTTsgW3PbVLaygz6it6WteyK2zZa6G+yxGJBLaOogzCjz0S6Nj2XxdZQXhvuFV77a8HZJjkC4FTfvgSnUwQPl2hn/aiwZAwanqGoPcOIqyX1crc98iQ694Q+tQrRDFB/hvUXPgmxtq2l/jGI6a2mErMEokZN74rOUJB+OEqPbXX/ZVHOhN1lUQDs0Mxg/DBWMvRR0eRUxX3X0Bz/Y+ua9teYjoqBv/6v12+zAFqC/q+Ip3iOMTyz0HVqr2BXZ9MNhn4doBjlzO224w++q7usa9rzfU/+q7t6OyBOJGqGCrkTEz/7Glbtv0/8klZRaYpCdU7Dm8x6rYiNGYrQWHvFVv6l1Gz8DQiNjhXGbhRsxWBgq9W1DoIXy0RXFI823cdCuRxPqrrfcG2qPRtg6dB9ihpj7Ss7w3+T/Lp6TLaZmHudqSbBrO9f/jRDZRYaMphdZIh3m09swt+7++rItikbBs/uo+N5AiLvy7KElVVMmDZo+gsnKwuvjJY3sch02LJ43jV1QPHDYyC5K5aJB2B96ClQgYlWFsOBFMhXDATiL7lSClTNrpUMGz0liMwhQhguNv9fbHg6Ydedt5kffvDgqfhoHRXO4Qk7w1KElIiyLh32Mlo2BCQeuqXwmBYAKo0fnZKlF945BaUBOJXNRMk3vZKki0HevLzJ4hyCHzvk9cj7hYUzYYm/ferO9Gzog3mJ5Xw7hAbTic0u+4sFbkxJAHwjlcdJlWFf9xSVUS19dVa1Pt+0fZiaGfY9AuVjZzf8rMoqNF8AhabjiPgIwhCfX+I67K/jzHwWGkdfm/azBkB5gECebimWHQGa8GKdQu1tBznNT8iXUBu2SxRQ86aucgNCbTekQ1LAFMQ6JB+xWTA4ztM8tghQuWLstYvXEOf11Uym1nK4rMhCT99pwv3oSBhDPpDDWKI859DDO/CaQQaje8jSvD3bu9FafT95tA1193+KHAVP58HLk+BJsEpPuGpOlU+cqXaKNtK5ICSSSOU9wflyMpCG9USIf0s5bNBQQxted0UAP0nS5FCPrC8OSp+d5A73ymor+eDIqoUcEudJrHII/FwsEGbBiQPRTX89DAtD3WXlB/z4QfTyYEtaeReWsGzsA+9evXqYTgsHG+zlE9E2q6hXpFJJfiBXhA16ASB4XuInarqiuaxVlbbxfa+XxQO06hDIebMQH+tUIf1Iv39VSkEM0S60ujdcaEpT9rdfUAEi3Q+HJKBDppKV5EeHJKq+YdL1p0hIQVc220DtoS2MeGl/J2UlVA25251diRgnySRcECgpEKlZA1GnCH/S2mNzaDoqYpUsf8JOEd9vxAaI5fNhE7ImAlPhkURoR2emrbm7vhleZNHczIlkb4J8Nm5rVLYTZIai3dc9zPzpB56j1qLBxuwvD0qIsqxLiqBz94/AnrH/ubRQUlOG4hU8b8MPoFcWdixKdvEFsKyNhVYM0Bv23SZ+OQxE2jLZaFKYngpem0R51BAVs49H8tK++hMAhoIBC3urHeu3jTFMNREak1Iax2YrKV3uEs2UZl5kURqCdj/U358SYoZpHCC00AlV/5OkVzdAIlGlNegboCCWxGevn77k3T26oLkU2UEIKi496oGHLkID38FVjos5ReLKOWUXbpWpS30Bkv5oUp6OO+Q57jvtImfhoe4pTn45rAAVMdmSUZMUO9c0r2f4EsOL1WpRvtSNyXfq1IcWVrMynOwEmPyC9rlYZg6rgfYtysopAZKUwWREZmqiqsW1CDcbPniCJBAiY0T2GWZNm5lqlDj0iuNIXHOcr1sryn1CaM0Z21FjvfJk0yq2JVKRAIR+Io6mWKphyLf0JVIEWU3qRBeHZzNgQlZMuZzebBccCbyh+NS2utyf1RMuV0ejQp1fpfxzpTj5cmokALZlKXgB6K61GZ4wUOvX++w5y5pmz2/aJEACLtH2e5R3FYveq0+9lDcfj7u9bPXa7XXa7XXa7Xba7Xjobittt3HkS+m5Z5NQvzudYsE+NxH4OGfXvbQ+7Q83XBou+yIOljrMbzkHSJfK6smF4ZrvF0ALmuFCOMxaZ6fQXvaH+rqCqhCkNy2v48udoyVetQRxVHJ8/g7HlWF/IEAABfiGxWP86CT+74S6jJF1apgf0Dz2uMlUt9QCMJvk1TcWyIKMneHZfSvDwdF4CiFN/OlmOJxscjloyFpMqt9mVQCi1oMDg4DAOgR3exkvjoqq+9Nb3oQhn4uGIGw+VdtRZctv2T39xUIgoVAaoSoVTrbJSCiUcImUfUxTboMJEA+g31DxbVeQFqatkKhhdQ0tmQ+XEES1swlYhg6VOSPVxFDhGD5aImoXoYfjMupdXjJhWFyQaEYVKfGRbs1e8mV1aL9dFzKXrXvq0AhcSSipobZh4DCOnQlT8pS1LcHhWIq7GPgTOqQOjTVGw4WQpWgyIdMxAm/bntfNDIcXuqHvMg//GnDZv9pWZerR5T+Ziz4sKhiUQnAwuin4ejH9B//+evTdqE0zv/ldO3HYYRG4iYp494hf7NKTHgbQdFP6EM0ANwgPl+lpzH8neof/toefvdXdCCO4GyCIUWkYi5W7oyW15Wi6w359oWo7KY0xZD51R+C7qYsZELwBfix/PsqHSggYxVqpTAbe18N1YBskczkv/1363qlu8bgwt6Lgyi+IPzPnxwTik7phOD73/479Y4xpH/8yTs/EMfZT97pmPBP//QR50OhJ//0Tx/rK1D87u9X6ZzsVQnBZh8de1Ns8LEH4aer9NcuNvv7J+3fQfhfpwq6P22oPj4bZ3MZ/Q7H+mrkHM0Go7vASAfdtvrdKXvocM/Mv//xlL2hh4gmIEvFQqRf/zUdqfmzEOnx/3WqflbDDlwFW0/FBdAMtlMW/q6GZu+Kf39Daz+Zv6uh2cz/8+9paG1gyMNX/8ff07h06cv//e9pUBA88vDV6AlhcEgPT7shdGuwkbf1atVNyb6f0x5S/ppNqdfXm25Kbgdvvil5labepvRPP0FvKmn4s1N2pZAh/poUvFW31t459LRbq7cjY2v9/pT9DLJ+dcoOeb3qpmC3/Lva5f8mQzvdLv+3G9ob7/J/k6Gdbpf/mwztFLv832Rcb7rL/00GRbv8DcvNi2EJqvxEyfN1H7Ns6lte3zBarot0IW4MsA/y4jC/6TB5DPVCdGDAWo+L7uENi4wGzHkBGKUXLQaEH5y3KEl82fpNtmubRgEX9j1h6hzAZ0CMQH0chOfaSBh8DO+rnxAqJTVP2e421U/d2HRKOyzD5byBzjCdSIZxkiqnzGWQywGe1n0/C4MfGGK5N/kB6wkBkCuESDnh9J3ckugEW/CW27/9+4HLJgceS/mPx2yS1BkvZRBec6UgBBqt/Fs+TsTzBZd0fx9YAg/xl7YZczJz9Gn/qqBgVFIuvEbPw0D+Kg+jFVzrIGaUUwghhTxkcPZd7ZNnovYIQ1mnTZdM6WwIwu9l9amIAdujYl1PjIKjciT9dnD56y7VsMzf7POMzMdvXO5D35NeBbHnsvqZqEAPWBpo+lEdFXUXGKqkfjwx2KXhJGXHRVM3i9T00g7Kgstw15Z9tVQ2yWas+QQWDHCNTxJwnnFMwZ3u7nyytHlUFSUFtE1SMiYi/u6vljak7G+ZQsHmRAE5gVf5+VDLrJGCzdIky4TltB4cI9TJispsQZ7F/pV236QhFtkNwg+Xt0FJRjDUS6QZ1PTehu2xd6HBxwB6AwA9i3gmzMfw8WoNqDpuNVOVm54Ntkr5grNmDphY9ORAZvD2DfGOvreC9BvcuN1oAvc0q5J4lVsw3sng1wPVk2BiGk/nxTLZjsgWSSwKNoUqecP3rZpB/4Y8XisYabSz2lNFQf9cfzrUiPYW+16GhWMxTSK5u9JqNKt4ntTWHA0HZdPs44UpOfhSp8kRROknP1odD87L+DhXwsZrHby/TrojfzQoLESJ5Q6i1ZYjS974DNFpIoNw8JZ1w73FG41rb3u1J2rLpzzHr2rwGUUFAFvsLFa6507YPy2HbmLoyxp6z/5P6lZfGgJMMHtT5DEpMHORQv0S3VaIGyYZN/GibMu/W0wqSFty2DTrRMg7NlOHpBgxP3f9EjnGRUtUlgc6gfJEagxbfokufGbTL9CpR30WRNRY/ZLKaygxk2ZKGhcFUEM5UirXvtZSI152Cs9mSzZL9MFsF5cUK2eHfEFV6262JHV2oiAZdceXHS7cyW5LcwN/TgY4QUhDVYFBpDOCXnqxR6bBa3QdqvfGqEzfVZeVVUWp7raNJDd/Oay0KA4gXQrUvus2Cx/2QiWjb3p4eD9BSM9PvSEM14qvm6TupWGDCy7PIsCz3DAIavD04zz9jQc1EKQFXE9riBJCkCZZT+sufwQ/wLs9SbYoAAgnZVVN0W1LRGCk98ZFcNqu9WXKdjbbZGtS3ewL4FSkwd0a5OJF7/fZpM5jdQUUEfF6Xwg5Xxh0a7k8nCcSS7WeLJEIwidL++io3w3JtlHPsjwYuagpphW0pb11rM9HG/BlF+ZB+HhZD64m4BFtgwVN6a8GpXnFD4SoRwanRYLwo+W9QGkaSO1HtM2ny+XdfXtUuMvfDMK9IXE8PskS0M2Qa17jm8FGSVZo8cEH0QlpJWZJTx1jcC6Z4ubt/Xqogfo4BweJ/CAMl7TvaP8yJKn0WggsFZC8M3hJWzAId1bt0dX1V2hiPqPB6UXFjxrrNgfXorZO7PD9kcTIWtT2scL32cqat/L1kHRc1HI3L2oxMr5WRh/bx/vp6IPLTbxdZHmy0u1AWm9ROaaKwYXCkDZv/w+GfCZSj+kPeauY/lrBDaPPtJhBv7Rj44Tf36eNv38p4qxyKeyBLtVVKldYsmRGvtxjFNO7yiwtUdVtcSsBWEH//eGgSFTkkRF8jebu9wmhGvEZ5aWMH7OJaGElszJ9K+NJfi7jqITGksEZyPgJC/WV7ifmBop8Vs+v2USo8KVRB2wOJH0U1f2Ma4Q8uLJOG6Ose4BVBnPbQ7+QpOOh0GL3/GIYia9lHvhltHKmpO76pcwxvfSLUBEuzBLT1lI6cVGyhGr8fLixin2FHOA8OtYNPhpuQBdx5J8MywPEOZ7/tOxnw7LiqBQR5P2rUXluBeZEVu6aU0T91FNE/bSniEF0pojBoSmyoYld+gnDa53POACpRvra5m+4+FXjd3f1TYdqXN5tQNeH4cYVTl1WKARCJus4KRwO1HtUZz+HA6Pzc6hgJnI2iYMDcLpbd1jYXzGbuU2M67hN8EKeJsYAbgMLTm6+L/GQL0b5Wbm7NcaHC973CSBbnZjhKsuFsnL3wVKh8ethturhkutpodHraSG43t1xKbjaMpGs3L23RASudHNQBi4yws3K3VvDXOi6vbyoOteano4KhXlIBuffEhmccErmmiUDI9CSmwbH+RhcljvpDRZeazGVbhPfpAcWrgVUeUT3PMaHbh6M8LtF5/ESKWMRWtYhLUrrgM3G5LxoALYZwISoFGrogyt3wcoRW2t3EN3cJ054XOedNh/hnxQY4faCuFsvOwFQee5olQoQb2rI2OYZsw5/lz0SZ9tiu7+gihhQMPEe1OxqHcrssGwBfTun+2NHxkxidUSd7pqZp7snIzIdkjjh/d9fRfZRT2h//6RHC8J+Z/v7PZJzB9L3QJyOqsODRV8o7Av1hgXEIHzok+zT7tikMuVZsdNmIAIQ+40xiS9s5vx4UiVxnPNu5CdLJILw2dI+WlBuMCi+i8AKD8MPV28W5zxcWdqRnJTGjITICEANmyQw9x/1JHu3OylFb4qgXI90GSlU+ARLWk+L6mZXWVlDOspWDZV3+1yXcqcvgjZwfoTI8jIIr/cloBJ1JRopDJ4GEm+LXxo9qzzUwwQBQdoc3F5ZaBXjVHI9K/zMIHww1rLL8R2Tau/h456U0qV9Y1CsILw33KoD2B6Waa9+05WppNGDl9u2/czltkgXZSo8ozfZQfhovHU7hiVy7Wjuu3KEamQ/kjGhtqegJ1RINmnAxMlgjRp+wZ2gIXXLI7WU3Q5m12UXDfgkpOf5Kk4Q3h1s0151WKS9cm9gmeRj40b28OTIRKYWNXbgGbzJ9kwOu/Xw5LDl2tF8MSKXyEM+PiCQCMLHS/toh7VctB1ZbxHqREcH5VmEzJbDc9SUakfxYV/KuwgCOQhv+6Xbaz50+aRugvIlVxYbXqJSPrZEpebT7U3zAybnybQee0BkLHVWoVGp9nK9PQcAtJm55wxKtH30Hi6BcLc9DPCHX2UsF3ueVwlkz6sk6eGrEb+92kc+/vbA5baDcGtAvr3gkEB7xd6bAIgT6T6l3kIPWj/LEF+6FXrSE0p962qUyiDsLXcoO3w9qukuoKrmSkLDuw8K5QWd0MamJApiwOzYR4ZSBwtnNR8XG17VtdjAImqyPau63Xp4Vbflhr9uJWeIfOoXGR+vGu7AM3FHOy42vAUpsVjyJaPREp4tqNfH8BbUEx3egjrR0UF5tiCz5ZKJqqTaUWz7pQZHEIS9tVS3GF5ttcTwaoISuKQs/XCjVEz96nYbWaICcVu1/aNBGTxNE2wn7lCUavFsdXkM7B4WR7+O9cwcCQUCWo1IgHabcLk1IlHAfQ53gfWdZBB+OCgB3idD3xZ59AbSsYiC8MmotK1nDj9fR5a+v3BFcet85pGkvXe8t25bWyaJkxHVJkv1dyRjARjnRVM/HRTBjsj9R7tNEAbLhWmTGH7qpmBvGRuUVaLDL8gQ7RbI4Qdli1tL16DkSv3hewrCTZBEDKoa/be1wqmUdbXRsmJRQ9KpllhvGVTyQtPx+XSf+S/ZBMsQmRQEwsySI0f0RU/0RSv64XLRaA6F6VKQftRJZzybcEuaKCk/FlWwVG6HBI0O02lmicFvLGwKStoSOZrG02n+B5RDMxdYp2ynesdb4lS3BfEwwSsAE25KQAyVsHBWBfkNr7bcBFyEUVMByvC1ljrn0oLHvdJy9ExI8tkFAsFdFBGfYHHA8yZBitISAIV1zSQkkqmiNrdNMhremxKR+lTWy7rLx4oSovTSReGXnxTyskmfJhmTzbRPE2WPVnloZSX6NLDbO7Qsia3nMinkhvFb2UKZgJos/x97bx0nt3E/fovlsJnOPrZ9a5+d293jOk4baJsmaVNIU0ii6lbaPfl2pbWkPUjblJmZmZmZU2ZmZmbG/J6ZEYykz4w2+fb3fP94nt6r8erzec9o8DOgAViBNpRlFOSyFjRxias1pSGHJlOT32SRAz7T6giLRMstuo6NjiH3oq8nuwAWtcoDGXkL3zhHrv7FZoPS4Ytg0Hwuui4s6x11wVxGHt/6ibVRjHtudDVA+s3ijpEGLzfN15VYU1pXaPBs8hiHakt08faG21j2PSqcSU1qdXuG10QfdqwonOl8aHK6eFZjnyIb3NCny4IGnwKINPvymngvArodchetjE9yb9nh1kiOd0uSry6bMyIzXI7jZIfxjeP7UglaV+4S04aiFdjh3lRpd5ZsvFcrINwgqCKOHbe7I1WThdC4eRlKpdEpZgGZIkFryAP71C6GfqQgj2ZWUHLi2ZWdRaKxajdiD6Pb1dfDBln+srMod82ODeD4Pqe9WXm40XUaZhs7GYBV2BlDh5vG3TkdOmcUL5reAymwdyOQJv5gjT0F3WLNYFGDNq+0OiaOxRBbjV/N0bNevMIM9EppoFeIZl9OY3e6qAIQj8cZyozfLB/ALEBK7PUwoMh4C7nEim1ZBS5r+7Oyru2ij7d2QLT7WFpUFphOcTArLC29ICwer/JJHJTxMgotSSj1CqcDM+BQYaS0+DT7KYYafWsDFVuzcqBAdxzfQwbTWK2RuA6y9SXOofpA66Fyl+jxuwcYSvRilkP8VpYSqkYdx2XYDaLhVUFCQBmZ3maMy+cgU4tffJipLpTQg6UoTroDpRgqo+WeQcWQwrA610bg9TjrYWMvII4s+25IhWzwIEMR2VCWu5UBSBHZsF0M3T5AntQUKCBU4YXeF6uhQOKCsgdQtHzbtjYmQA25/wqflNA10Y3ibgClaQRCuWD5+XJJ9juhhh+fX1EIEr6aCNUvWINPIgA1UPMfqXAhyalIvImrUVCVKYGwa6wayKtIsuFaUHBGdCh+DBXezAeroJ5KrMMhGWHoyM0jaFCQKxp2pxtuQKlNFFATSzR0Hc4Vb8vfAHtuSI7eBIhRnAFf8OsBOdSwYzmOZ17hBJExz/uEFHiTHCCH+i1YUTCHwxwIv3aIA6Dk43kAxhQBUFtAFDgNcp0bHJsc3Vg2HZeEcDekQVm1A1SAONDGEEV6eSgYABy0A5CmkNIjPApHZJhHoLTmeoHTdJRL4MQF44Ed7wI1lrUXlEM5BbV2qbEEe6mUGuUZWwtkEaXFGcFW44ANsdU4MttzeizM1Hc8UKO7K2iPrRHYeNe4gc99p4d3Xd9uGF2v7TQ2yHRG1H+M5vrxPEZUHLu234znSUk6RUNqNJhGa7omKtGsjtvrmHTN63h2/H0zJxgtAPHESotc3ItXiIEI+u9dGTp8ryI0D4LI8rtzr2J4a9mrTqMwF9m/vwdi2LLbxvqq7aPJTzzRQWYYySq3gyBVkA1DGJ7Mjv4Zh4Boehu9DkPn0tCauZrOk1l2gw4cWRoY9M1HkVks4Tnaw3yndFSP8dF8pI/S+Cqax40vXKwBUe6TjiI8y6WZuhmes2ipRHKfemKwb5Ireg0C2xWdrNyYRyBJ22W0N5dH57NgkIaj4zyiCTMrMNx9PPWBrLK94thAtpVQUGVLqIJsGMKYlS0G8pHOJHw4Q5Yw+0DYy8Eo+JMsEBIfYsD5YI5kOSB8HCIK2GCByDyP5tX4xcdDv2efiB7GYKRptoOEGc4z+ahkCgE+QAuKTgkFFZWEKsgmIIx8ZOyEXernMBOEylQMcMsUuvahr6JCg5B4jAHTwTvEYPIhrIAcZGn7IKPIHGOSoHychdPxgVMSsF35iC+17XYHiE4pF0XmMIMDpKMwSkfjIIzwI4EON6gZUN0o5cBIpBwgHYVRdiQSJB+JLNbx2u1OHYhDGRZFYQLGisIREKTDfwAk8sE/kqXMdjtcdtwVsEnoi40iUuOwDM0k20nRNB8tgzNG+jCbzifI5gxKtlVRArQ4wPXQyXTV0lhnWYbmINsJnZuH2Ri/WtloezRaa1ZSrQAOrFYpB0hHYZRdrRIkH4mMucQrUoAYgFBeNARAdIDGAD23/cArWdAeEaj9OMYkQXkhBjHObBVoqDTRZvtJtFkgcIVEmy1JtNliaHZkGDQFgEbyWZeuBVWjUYDJSQaLCB0+wAduv8y3m2YjLOtxJVRBNgxhzI5UDOSDlEkblMdA9nEZKP0iJicZLCLM9CNqrpmJt0yVjUYpLr9vAk33UOPKm+YsGVgeZjkrShko1E8uJAeFQnatgESrV9v2ZB8YGWU2m+503zBeLEhOvDh6kxzxEiGzBra/oONVjShLKn3Ay41o1WA/ZMPrLDmuzUhmrsHG54yttjseNPo+xiRB+TgLZxpsGuJ2HiOwrPMIYFDnMcWKwhEQZHYeEyIf/ELhqeMT2iBpMWxudKAib4xIg5D4CAOG6nDBcNIs1KssMrnqOdM/TdXPYoHjuiqY2QgvNcd1uLwX8wwu7R3PLli9eqGkF0smWM6LtQZ4Y7GI5BIFzWYvea5dzMmSOQnMzOTSA5rnAkFmwZyBUw6JiwVzJk270oI5U0jnYsHMMKUFk0lzCybXVbHAzfRp42agRAPyoY85qwwY/VPjAIyoA9nFdFIsy2WzAri33p/FrkEWuwZY7Fqpxa5BFruQaTW4VpRyLGtSA+tEDbImtXJrUuvPmkDTcyCUFw0BEHOYwxh0ZfKQOqeUb2lAELI0NAiJxxgwc5qUYvJRyWRRF5xQ5CFR8IeLSFawvwDQgR0paLn5jw9jKRvmFiCokMRQXjQEQMxCEum5g03MQJOEJRQ0JE2oggzEihNph9hYZgqtmFRAq1uMADRHUkIxo1kDolkbhjA6e+ocIB6WISG956jWrxvShiAnxfQBbEU25ssObHxLKDB9YqogG8uJoHLHZaA5hYjJScaLSLG85SMHl7Z8iKBixGUYoc4XoWWnNlhEmDMhRM0dBnVNt9UjezCAMPeDRkGfYqOw4gDTAR2fCpPKRyvbDJmW1bbx7RegeIwBs5uhlOHad8/HKwf49r0AQfY9hvKiIQBi2vdIz+3veV3bhT/HlWFQtzDFisIREGR2CxOCO12CKNMxYNvUBwlNrGRIUD7OwpkTKzTEL0L5qR+wCDHmh0AoLxoCIHYRgueCCkyZycszkMmLmJxksIgwTR5Rl6YvZKD5ECt9a8Xw1g4AULFpOcik2G1LhHGXZ7lex2kYS7YfMipFnzQ0IV6gmboKrCqrozAJ1dEMCcozBty1PVYASjlo9EhxgHQURpmjxxTJZ202+e2OF/qeaywDyZ/RTfGcQfMUkzwHXAOcwqWZC5Jg5tIkKB9n4XQaM/3MdI6bzahTPN0XjnaduHhtcLzhH3bEncyIwfLiWOTg4phwgHQURjnFMUb4kTDdJdtplc7IQBwYiZQDpKMwyo5EguQjsSWDGcGauT0rIWHbkxWSOTW8fXkboNmZleFzzLxeCIkdt7u1KN6RFTXaAXKfqUDpvSplVQ0moaqWIUH5OAtn9nVoiLuEJ3vPJn8JD4uFFrPkWIbmINsJHbXDbIw7JdXpQq0ND4GmpAiSFewvAMwpKazlDgHRf32X1Tj2g0JDwCwKKw4wHTBnRCAqPs+HLJ3GK6srTDfcrlMH3cptto02Wj89XdZ1YtNQ16lAM3UTPGd0ykzyQH6eExgOxvQhQAhNxJRykJGnOEBah1GyegKfWoiOAaUf8XFY5W6SlRpT/dD00yjsgNnupAi325ReS1Nqy0EStOU0CcrHWTgdnakSiDrxqROYRtOcYDnoLwVKVzbDJDcFaowUqMEpwFvDRkN9RWiq2m+EpqD1ZGCEpqpwhKaqYISmqn1EaApYZ3YwDza6pXM+KVYUHgJkfdiSIgfbkoQDpKMwyqm4McKdP0DXxZXOHxQgaP4ghvKiQwAEjdmGWBw0dRDpuVtXOnZookuy0NlcxtKG4biWvb67QKzYaPusvx9UxM52FrT4pqNsm2+32z2oeB0EqYLsGIQlyZOZbUGCc/l4ZtoFScBQlHyTymLsb1Ixxy9qJj5PuaSo5SGwqEVQXjQEQOwiRPQlQe4smeVBzkFwkAmUFw0BEHMOmZy4V/bVqkiB5TCmCrKsUTJRL5S1Ua0fFOxPZ1BYcYDpgPlJJUNx1wqgG+z62ilFg5B4jAEzv71QDHe8hYKFoOjGOGPNdlrL2bXOMUIffjcCASSK7Ta+M+g4SGBZdP8HeRUoPMh2TMvAmOBjTtBFPouLy2ZQhpiWBcaFHLZidrvokDUwOOSKEQxEdxiOsTCc9mg7PD4CnM3go82L6iZaVB+5n4AAy3PxeXQh3ksP1NUYbCzbjZU4qS2nE1Qay6Z/4sSJEwcnh41L7mDc+pLLLj48mf3gjktrZSjvX3qTOD6RZC9bn6mT5DfwmZNSVJgOuGYU/y6zkDGUFx0AoJLvDxmK/f0hwrgLBfBvaPBcQkEmN6EKMhAraZuzGLttjjmuPWy3TctkfGEpByHDSYOQOJ8TMcw0nBRTVtis0ja7ADFKpFUskVa2zY4gZjcj0vMLmdNxwtLVKEUKLGQxVZANQxgd7nEI4H4qRXc/l30qzTPQp9KIyUnGi0jJmhYaytQI4G3csWK72WGszizDoCFlihWF+fDDyy7zKVq6JifPMFK9VgzTVBEhY4j0KIns80y5A2BSc4HORDq1hhueG4QHJ6mD6KOJZKKo5IsF+D16MZP9N8n/Qo70VaLKVpy1zVbPhexSCQXW8ZgqyIYhjF3HI4D7XWzFaztLvgPNApVy0DQHxQHSURhlTnOkCHeKeMXpONF687JhDAOFhjFZFFYcYDpgDmMyFNdMYXKldM87gEFmKsWKwjEQjKLg2wHaTTbCY4qfy/JEdKp8O1oHOlnO4put8QKNPjxesSKPizmyUja5u1Izlj3fuQ78xN0HCc2FZkhQPs7C6SStlkDRDQBeD93Vja+Mn2C54H7cPOm4JlkPswqVt75Y6ONmjmVoWN5D7WBfbElQasygZKfcT/YzQVSAoM7myeIE0cnCBNHJ4gTRGKDn9h5Omg60Q4LLQL2HiMlJBosIHVrAB25nHiGl09MFCE5fAuVFQwDESV+s5w6lyDXVbbD7NckCIfEYA2aOkCiGm6jLG8YqNKTlQ1CixlBeNARAzCY8q082HXd9+1A/mBeE/XCoz3m0jEtXQ1mBWembHgPI0hyALCMI5UWF2MYtHe+DCs1BJTvSc1vB5Z670TNdA9wn3QcJtYIZEpSPs3A6LochCB7A9YNCnb0sCisOMB0wO3sZKp/80xCJL8mBg4BV2fLhd4wwt9sb6r1DHNR7pzhAOgqjzG+3BYQkQBCajZWDMJhPosyIB13SE191Fl9+sSsPRAffFeSWXTwmr+V7q3Yf2zQhDko+igOkozDKtJwpwm3xW74HLbHjMlCLHzE5yWARYbb4RJ0P7EIWMV0HXfUSXeh9PVuZWyyTd8lf78SmofVOBZodKJ4zOmFqfYC5veXzfTvJLeud5DnkDpxTGOrm9oNCtjSLwooDTAd0Kp5bRvG257P5XPpVmK76Srw+WpcsCivA8hvf9txfaS/SvNKe0EwdWNpjZ3Q+gSUwBrlzHBFcNnmRYkXhCAiyzWo3dG1vvWxWBcDAgCVYUTgCgszv/wnBN/fdsHTqN2JyksEiwjblWM3POXQGcADZjTIMTMcEKworIAh1ikfYJJjiMcEd8rXanRmDcabOHAssOYPrJrpLDuGaZLoDxEcYMJRyYzw2+ucon8lNxPVFJ6daHe6HJuNGbrRyY8e+AoHuII3vuQRp7gx6q92ZZgxGSjmwE5lygJSBQlk6ykFBw5ggfAPU7kDTC1wGNFKEyUkGiwjbSGE193BEtMQD37Z4LCvtxGsb0G1+QEjn+ThbWeE6ZNemPJmP1bkcGppC65uHjvIHeI72MN8pHWdesubPgcBn46HbqlDN5GVf2YQYxssmxAoQNGcTQ3nREAAx52IifXkGTxtmEDhBaLrQ+L5vnpnBWZ6jLWZwxikd02I+ZVDuN1SCl31DLVLQN9SEKsjqEBZFoet7J9Elz0mxI61wdpNTzg3+Z4oDAKV5olIsVoAlK8aoDn2hKsPA7laCFYXTIFiSRCM8R8V5sQJh9oLAMdHVvN2VKo8Ek5Prt9kOe13UcbCcRtgH2fB8326ExVJXB76iAVRfZbMOJT2I4dewF4plMfZCsZgrj0DpEuQixYwm1HTUitUp10wUAw7Y+MzOWLxWM+Os2XZAu8mHIKsfQ3nREAAxrX6k5w40mma7gdbPQIsxykHoIxQNQuIxBsz8CEUx3IJEuLKCVKSggpRQBdkwhDELUgxwv4LY6yZa1dvHES0wCX0FyZCgfJyFM+fWaYibEwQsy4kiBeVEQhVkwxDGzIkY4DZ4BIL6A2UY1OClWFF4CASLdhf2EDC8QNICzXv2tT3f6+fMD4iDRpEUB0hHYZQ5NEyRfCQy89y27zr2tDEDFHVKc5DthPnRL4vxUzJCS1MS4MCUTDlAOgqj7JRMEH6tNVttu/QgwyIF1tqYKshArFjuZ0qxtFOGD42w3cZE5RDbFbuXEnPcAZ3l2ya0Ew+E8qIhAGK22pGe22pbXhjUXQ9c4V0OQq02DULiMQbMbLUphp+wXhhAnQ8+BKZ+BOVFQwDETn2i524DtZptM1g2AruNb2wkt0kfKBJlValIQVUpoQqyExAWTcVBxyLkpIfZzq0gWZqCf0/y0WrmAQx9SQ0HsdIannXFruExx7Xhlm13A9uGvmuXcpANpzhAOgqjTBueInzDEGFQF2aSBULiIwy4dB4/z0b/cP3LTaCzAppnO866HfTrMVqtVeuHTT/Uor1uYNYDHasKyJUedACTUIc+Q4LycRZeng/VXBrYLrqIfKovNkmv6SpcEKqLi8Q/tCaGUVwBkwumU730LAyY5KVovQanaL0GFsQ647QBMPnrvIMzaIi7UjoGa+TEVu7yZBYLLU/OsQwNGGYoH8pBsOdBgZAYjt1NsES8Ba4Uk8+BbLeBrOTCtziGBr4NOPeuhlP2gSrPQB+oIiYnGSwizA9URM39mmYt+dBSBC4DBpYwOclgEWEHFqvzgc0c/tbwbTPM7jBoeJ2OiS6QLyuDIAiVQRqExGMMmFmuKIZr1xresu3bjI2BfZCQXcuQoHyShZfatQIM2TUa4k63RGDZdAuAQdMtKVYUjoAgHYMDIMEdMDe81ip4MHQJBfXyE6ogG4Yw5jRXDHB7iciWBct2G1p1XQ7CFSgFIfEYA+ZUoITh2rJG2+mW2bI8A9myiMlJAIRfItp2M7fAFIkmKmNFqiw8mMlJjhQR5lXuU+Vs9rkPz5M1QYNFlvltMqOmF09SV/ZU+nYApGS2q2+bVtnr8YvRuLJh9gKzPVEBkp5vuJbNsFV+JjyAgYYrwYrCERBkG66Y4Nf/ZbNjt23wY0k5CNZ/CoTEYwyYXf9Thjt9tNT2PCgf+BA0fRRDedEQADGnjyI910gsOaFrQ5PuJRTUbCRUQTYMYcxmIwa4tpbxrYDLQLat+I0ASQaLCDuZc5OxcEjACdtRgMlJBosIswe7BE7W7ssgZhD32fCysApDGUXCWDYts2P6Vu4EoSLJeksQmmHAchslQy4QTttxW330QmES6oVmSFA+zsKZHUsa4i6TT0Hoe0Y/KLSiPovCiqNMB1Df+gCfhlaxZKg+EwHqYfeD8hOhxkqEGiNatb6iBfS6D+XIxnLPLD3yneIA6SiMMudfU4Rr203W5eYHQaogG4YwptU2GRec54LUCJ1GWXNTpKDmJqEKsmEI4wScANym3fQbdunpjgUIatpjKC8aAiBmmxPpuf1Cs2v7YQ/aDF+GQf3CFCsKR0CQ2S9MCH6KN/u4WqcAgSneLBh+LBoCIHaKN0FLP0Axi4vRCXdBt+2EE5U9GR0W4oeJyu6sBn+7M5Z6zTDrhD75cFdG43rowEOvMVHJBsA1Wt0eGaAE+deg/bOrTsMOsiNAfFxhfDaft2r7vmMhZm+GITNfRsfueP5G1l+SJDg1sgqcSuT4wNizjtldXOy5HbNrNH2z1bHd5NIFrEp/Z8To4wf5b+xR22ukH7PQw0RlS0aFDlLcnpG0fG/NCL04wXBMcPekF8RXgG/L6uxTht/ZXZB1vcDomOu3yih8u9HzfXReS8NzyV7e6/nAROVgmQ8uxibLMLTIv7dk4j2jh/uBA+c6O/18yUeNdl9e+oFxXZ9e+n166RqoEPeFooF2H283u932RnqYZY5bXESlwTbWfLwj0Q7NvkB06uhYHyAjdDGDzgXFfo2Wc4NMBBdZdmji0uu45Yy5PsxjVmw7OYMBBCxnlRvOBt+5aVlMfde3u6ZvM4rG4mIkWfJtc8Xy1tyJyjEWCsuZqYPPZ+11LTNdhQIzzV6bmsuBGVxl97OQRts2/YszWrKj1HCCNZNlZgBkonJBuS9lxAwL4ARgolJju2JpDrCdUIlfRpHkHy+hcAYcAiFrPW+Gb1nKlQCHSz1Apsyx1vMWCkIjUwbH0FrPWJ6xEgiZlDLG8r1cfc0w2O6McPXIYAxzCctZ5b+iUeKBaVlHmEDRKEwxWYaCndpU0SyBSMlkJ3daMAeZDDYNUFLROQolVSagHIAEEn5DGsCtGT0O1M6syHSNYNlphnsz4qVeM1hctDvdttmw9wEq3LdFv+M+nbked17PIpK21zICO+4vo6eG2W4vmY0V9HAXO+7KrawaDbvdDvC3iJ15mW+jntBCKjYbyza+JyVvXSFlYt5hl8RqzPIQjt88Z3R/b4wHRkaiUmTAiCU5niUpSzLI1qMitxdUY9sA+5z2R/h6c30fS4/MygBLaTmrzDA12M5MyzoI6YomBMSKMjB2VFVk60lNBNM9rYhAISClJKmAyWg0y+A6e4ucBuxegNqJyjzXLUd5BNDBL0tOp8+xRUsGARlLBgE4AcdBfS4FD+UgtMDU8M01RkeBw5UAR0s9oC3AVF9017dXyVqhIJ/0sAPXWFlNOnAcNlkYi05rMHwvrN9EJ461HuTNXdFNZMryJR2DufTPl8ksw1Hm7WTWIR4N4+mCYrUvkr651g/Wdqx+sOWGOVFZ4WINr9M1um3TPTwchNbi4ipeD348dXPs2LETi4uGEbW7Bm4sg7a3ZnTNcDkH5itE9l1RZlwAQ51uWWqnBFhYsl4UCli+tpMF4lQrla/tNICaqbzdj/S4ndrP1KGGJt8KpVrLWWV72+A4NC0LMixQYwPVEjCJ4RTiGEwKgA0mBWCzk29ViR43JxcWVGZJk5InIBOa96MEKJYrk9HExBrIkJqZsWpBUawkZrFtKoEywwEWRCw9k8m1U/n+rMXrz1p0fzbfWbU4nVVKV7SbcDpDxYrfv7T4/UuL3b+0SvqXVkn/0uL1Ly1O/9Ji9y8tTv/S6q9/aQH9S6vYv7RK+pdWSf/SyvYvi0YxKnWdtjlRYWpx4wYU7dK+qZX0TfMByzX1+coFdpDy5pXVL8oXY7AFWkUt0OH+SNSz6cvTlb49Jd2lfHbmG+aceS9ObHMBNAk9wAGGYB2ewEae8/TI771s/XhB1fbIFzDqJw/yurZvhp5/3rVM6L/SP9pZ8B5boWKypRaIrTPjexuzOmR5dkEKy1kF39+AcdOyRvPyoqUZzCPZ54IPK6vkO2P66xoW8l9J8ULyUeYN1hHTVihtzE4M/W2sUIb5Fmtxsem4Fgp7uCOvwXasUOHiiwu6NvEyGpQ5XlQJl70gba0z0olKnkWXnRVZJE1uzMux5Df56LuXTYzmVFav09nIuh7gIJVEh+t+PkqJMGlCsqTlBA289CznTz66iTDJtiyJf5LQ7mbpcw7JZCMVhChXSblCF8zhb9xR0sUnTC2i++Y7KE+TkCQqF1d1dGQP2j+afJun9NFRPgMFBd5w6vVCyFOsw7tS3QbTreN2k0RMdE5gROpVo3UKXSXrO+YSSsRdRRK381cROdo91zF9I/SdVsv2UV1FB/NDlRyhJZX8xETljv34axg9F6W+Y7ad62yLLH9Alt5oeN2N45jBMvS+oayPvo2PbGqYruUgcxEvD4n1XdMP8E2T6F/0ib9nu414jARDfq8dWx0Q2JXV4XKD1i/k3JjWquk2bHIk+O6crtGwu6HRWI7rGLZOZqfbti3SYzGccHlzqssszkCC5IS4AEVqEFQkZ1htS9WNcN3wwmXb30HJ8Op0vFr9dCJt+rYdmcGm53fMMF5EEiybXTuqGmR3Ol7jntwmGflKW6JevbaVloY2Mt0ZEbHLO2kRrqq9ei0p+JQYTR713MBs2hkn6UqSKIfsoGF2bWNt2QntoGs24v1I6UQE7p3EzWFObLJgVrOHyL6avRiM8teycY4718XfS6Iun2HZTbPXDo3MguKkP0h6VqgTEJirxLSzEbyKCkQay3a0wgeZR1TLRkAEVUXDXkfru4L9OaLrew07CKJeed691wtR99a3A6+9ahu+t5Z3nxK2v5pLhTh3ose8EpffaE1bPB5JlPZ66JuNMHOY20COaZjdsOfjYAVnx7pOh1rkhfY6GLj3bYQ4X0M7Kr1x7x0t4ooWJOEUMFDuoTp0z5xwcRFnNVR6ENNX6YnBUdpvnHH+4iLJgTglhzkIXoA2yAFMy9oPqclit8A+NcTW2qd6ZnsE0qP88nroU3iQjGWKhGv0AjsdYEIAKTNBYklQWtmuhc1xToRyIrJZ0c0KmRvW92VVbc83U6O+i6GE5Og9R7JyfA0vVqKNQGhZPt7xnxmUFtkCeQBd/IiMtWuYPcvxjK7TtdGOJlTHbXQAoNfphqMlVAttJmYj8YaugxwGN1FofV/HHi7DRtgAKp8hmvRkEmgOwHACg0xe1dggSzPOdhKVwW4v5ASx2e4FeIKBTaBhAcmficrOZMCO20LcjwxCf+fJwHMNZEHRfdRe3AEgYlyQ7SC53iw4dtJxT5qLi/jJ6LmW3XRc1CNYXLy+IJuoDLNxFIiJygQbQO9G5it0UOjH2SC6xHvVbNtuuC8DhT1UEpJXDWSVHoo8vsba7vrb8zoU+11ZYfJjZ15O/N+bFyezAecxVSPn7c6rXM/FtqnwFlTlVmxAHKAtQ3lxNvUG8moziKI/UdkD6FCaTFTGAI3nW7ZvW4a3hHq1E5VC+M0A3ZUFJIcZGM22Z4bwG5c8rw07Mn3f3JioHMioorwL88/jDKrnOqd6Njl2o7KfAZF8ZL0om6IjDCotjKMMgkr6YTZCcoADkATL1h+SKVRljAX5WKVg9nmcQXGSL4VI8rG8cALsnAzyBhkQWgy+Ym+wAsvLgpRKs2APgzBzmUNp0sw5zEbylWCYjZJsytY/VMXTBBsDddmoDoFMGtFhUE9F5mAGWFkz/RZVTKLnfFgSLJP9+2CGRGYYVqYhzWYa2mGQ938vRBDfRyEVL6UihEqIbPFs9twGlQ7kMV+yYoh+GgWJTDwGQARqi2IdsfBDoC5NwUFQn5iMbE5ji0vHkDznczrBMo8MhlMaEoZEcxBWxk3cOKzm1fIEorJ0iEmQypd9DZJRCUIe80UrhjgZGiNQJY512ZgMgwwVkawnuN0rSe+EIaEYgpXdXrCMhxqMF2RM8zDMpOVvNwyYuaqdKlg5RRMkpxjvTjsBkxHQc00fj/V9OyDnMVyfF6WlrwivOoETDjHV2X5AUW+v241eaOPRV1zdSGez7YS2b7YXF6/PPKfZlsOyZi2npF8T90VDvEoVlyhc9c+K5R7qscZNVmgHYTZ1cpK0ES+gOG0GWdpsG19Q00GeSiC/mFtFYZr7kAMcqhEOQAJ2iEPQYbsq4qIeVdf0w+yMR6ovmfHIghNZf9PikBWkyZ8Hs9Ykr6WjMJplTnqOa7gemRKw0dY229qbQRYXe92u7TfMgBqsxKrQCds2UW3PqdC/3R2A0G3lfWl7a/ELBnKqhtl1Qjx7PlHZlejMEAc1V+ooOR3jeOwZdPEca6Y8FWSpsQFw/LZhtp7kQmxsgrbTsHOvy4lSY1OEM8amqM4am6IeMjZBtBgwSqTrM89U2mcxHJB9sC5riHJKMBPIttFsquRlVCYU8WwmFPUkSAfZAB2suMb4PTd0OmgGxUSfEAyqgge78kzkQWyofDtc9r0117DX0bcPEqeicKJylOOgKIuLkW86gW1lPc+JJioVJpyXxBMBXd9DUzZxeu3Li+lU2horbXPFCLye37DjF5LvR/QL85LRDJl+r6KKSg1EOuiDj09lnWH6rR7iqXmdnBOz4Xupv5Mg1PZaTsNsG56fKXIHQRgL6UpyiIPR3o2AnNNM/YITpun5KTIAIqbfClJTnNO5GxOVnZAqrd+x2AnSN+0r6hwLnUradGx/OKdEH0saIQXsAIHdtJRMHEaTipEiGvGmzV1WkDZ3eTDb3OW1UHMXMVBzd4sIyYnpMs1U7U/cet2iXYyNXqEkLy5eX5ClRg/AM0YP0GeNHgDQaRJ393K1BQUqK0kzoIBmGt6CNtvdK6jpwMT1qW2vZ+1ITrCT5qKvTc519o6sGP8zUclJncBY83xrd0HquKHdsv0LIsWKvYG4xM5kc4yjTQ0w1wucaIf7IUkCTvWD0okZZ0gUr3yFGWepaT8SqJkpGvRjanyyEI7fAKgiARgBdfTb4zKe2hb07uQhnQSmgewoltbQXm+L9GSRp4sWILa3UzIzDH1nqRfacQoAfQGy1TlwVu3YYOKJFGSRo/Fy0/dSu51XokHA5liH7CEafSUwbflz3a6cLtvtyinpOCfu8ewGY/yYU9Lu4xa06bRDO/OW6/OitC9bhDN92aI625ct6qGOZERlamhBlhYKAM/YVECftakAQAcrLkjxygY8FREXd/RJ3nHxTHwuzhUOETgtF+XH2rJJ9TsgEkof9KkfmQo6ywqydPAM4CSEce+3YXYDI8RjY/yyjJyc+YJPTTuHlrfspBOAnpHZTiZ1JhJFu50NZUaQdk3zIM6//QxltoeQ10KtIWYyxSknSVvDApppDQvarPEtqCEDjpe4FuNzkKXOlpVxFka/Kq4ZS23TXQHCGze4S+hjMF6HYoe2n3zJXXKKszMFWVr/ADxT/wB9LspFABriksm/tEeZeU7LUg7L2sOckn7NOc6pqhF4Pjq8pd21/THHDeOlfmhpH1qtZRhB0DDd5vFVz7EOHz4xCTJmgCtN83jbc1vDltdbaqOZoBLYccPJYeyiXzjx+SAfxpx98MRpCbbZ6bSMEN9kTbacn50KGr7XPWcZ3+GybNZmZo1le/3MZdO12mjcZjnmGa2eY5FPj90t5LdlN9qmjxd/XEYk6Iji7PxZKi+ZP0vBE4Mt/NEXLU4gpzRmn8/Bj765Fg1EZvFzJ2gZrhk6q5EhRQuLIpeQaqJyMNHhFWdZPBZNVGqJDO+ntAPUGGThrGKiMpxobN/3fIrEzxOVzS3yTWqp1yTrBHdhAUlutDLJ9skC0z1ZuWUnGpJGpu8fT1ObpDBOk5x6yQycBqXemahxVyEWb261ek1jZXVxMfoxlBcc77nIMtnWcIBXuLL17RK9U6JvwPpYu2wi/a6iftnzwxMDBTmuYug/J3YUdI4bAj6RlNldkJN6dWJnQYE+XZy4LHrK1QIinCSpxqkF2dzaht2lq73RqsjtWIaKXLTOFa1l3ZoI3fiAurMo0crqHvyE3OOV5bafHFRxeqvVaRvrjt3uHcI/1+wldHpb5HvH9lu2ZaA2A6+RMd2WfYzmgmXsXdtZQikByev94NHJbvGSpUmmG7Sjqhsga5fAh7kwbmVi9CgPDXodvLayTzroGKhZ6J9ueO5qQrMTEdF4+WnoO3167jVDfDpJfzQyV3Q0J8rovhI6myfcEKAj8FzP7yR0hU93U3+rXLITEH87vXbi4gjXhd21zTSCUzz2VM90Q+c62zg131/SdU2rP59RcNHK9VW7kThg1xrKgWNl3Jzbp5uYn+2Tj8bNfeVB7KxpBmF/eeB0ag2v3V+Sttq9/opYK1/Ip7m0iSYBLbsdmoZLFXiGRSOOqE0FKBN82+o17P7ygnLaV35T/FJ7pb9kbfQ6yKD1layN7kafnnruas3qz4hFrLXWt9cNs88iE40d+mot0GwN6t/HcK0fGDd7/YXF9Fu06eWzlkXXv+00GxvFnbQwrYSZ/OuYK/HON9w0Gy3f66HdV6G/MUCDuYK9l9Zly+8BhgrNI3jo0Fa0zDbYjKlVx16LXr+LCOwGuibSCE/VjClkHy/Ny6uRfGdGfmqeiKdy4lnjUoieYYmrSFwtiEG/p2FPpmFPpmFP6rAnNZYY9CRKk6lsWrmrze40IO+sx/LdGblzatpYD/Briwq3jV3sySnqxnrkZFdBQ+R5F7XERf4lNYPhVc2A5VWmvIPlWzLypWZ1Nko3s433zeGagnZwn4nFUW9wK3nAdsHr4l7x2ViEx414tJs+Ii2pJ+Ey6g930RossjUHd7e35XWuvUYqK3VCMTrtldSHkOz3QNTWjAD/Q+ITrDmo8cKbTjYRSW8pUsXdN3s9PCsjifROu4f2O6ABQyRB3fro8OUzEknVIq5R+qAInx499VzrnOhn10YmJXTOSp/t9fC05ClygvY8kDRIOj2N5Z67QpyhTTFd34ve5nXJBqIzoyecuMQfF7XA5NVuYjHOjp7Jt7eApEWn1ybunQBbly3xQ/x+iwSHzFCQ8bYdnh3JfCtYc4Llzemj0+p4TuSG7BjDuzR6jjVGydBeuR7edtbwXAuVHJLlJKvxGAuVltA3nTA4KxGiCJIkazrtNglqFB2c7iRKVq8b/XBWiYcNZFm79RraV+DUazsKQvQvKUB4qFAljefyvkhEAtv0/DXTt/DooFmvkYpKtnfFKrJGgqTOUhsNMJ3ropKDqlP8qiEiibYexU3WhttY9j30KW4c1KcjTxOxwxDk260kpSZ4ANnhbDS8nhuOloO7GMhuSI5ePgIpMpkPOkXl+SCkiMIb9LpdD3098rpg9PIYOlQdTM0IJEeAeN1grARCcTpQwpANuIdLKMsOGr6Dv8SWoUu9ZtP2cV3gRcJeRRXAtdd4ASQQSl+w9JFyjE3UHqYeTHIqkElKTfYDIsOL6sexfmByJCjCj/aHOy0XL1wpo5OzRpu2D6ZfBKfVj5cKHbuTgjzv0nkksORRFJp5BatExKAsjX6DJiHCyDEUIEDtiswWM7KrF0XIRvsg45mrIQCKdpPiHIL0ZD6s4XUdO9gF6FEsDmXkvt1CV4RvoIla9Mv2Y9UYg8Nzx5H4GIMBxROVe+blpEufn0yMmbIp9Qy4JeN30ZDgSgfk5hCbwpl5kK2nC/X+LIY2tVJNzSigzTU0ANLKIXthBMV1Z1GF4jheFBdN/MFyCBn4A0wsOusFK4Bo5BuBES6CYjPGJUgDcJTLoLNNaAM0waWpxoIPUj5ewATTHfElxG4WMF1UcDyNVC7HUSxteG7Taf1X6htQHGiDj/ZaeEEIlC2oJTtcjsXt2GQ5mrZiR/qBozbsMj7LyQAAKwklbTkMtyT2NAzUjELLOchk8KiCnSWZZhUwHRGGqzvUqtJQqw+PKCsM1KbYCJeme0Swo43teLa/RUzWBk7gbMuBApX1ivTpKGu+B1D7dsPzrZ2AxrXXdgFi9J59Gbllr9KWaGdBiYcRBwriJDW9DrF73dDP+tyIBjbo5CTfbg6BynQ6YBDUuxGR68THanMJTSzGnZfJAoMuE2msgLV/qh+YrgWVEgdpKT5cQlJluZgq2cHZdkg/XBCS0Rc6Ihhn8SgPIF8Fi+lJI2QYV8ySzChvpKgmrW0DzcmiBSj7QaJpm+jIlWBHQYvCXgxYsfNwoJQpDg4piu46FCOR7zkUE7PQLTjGQ/ApNFHWB7hYHeLhVGXkclQxLZb7iIMqabEYswp8sTpTZDFVCtWgmEuQyWd6lBrzYnnPjaGK6U/HvBvmqn6xiBXbBmaw0GcdEqxi6aZNf/Ylcc0PMp35IZBJrcNulj7bDyWZho72iu92xfFGdYCcedgvPcgHyVwg/t5CJk43xYI59GNxsdHtoSEdsnXJjXlkou/4ErqpDc+bkw4foW55k93lZuz6fvGscenNenHi7ua+eOZmvnjmf/ri6Zv54un/+YtvXh5P/0/zuHYzY1z7n8a4ejNjXP0fxhh9vro5L07d3cwXo89gN+fFqbvci6uAB3TDiQ344iLVB4CCCrigpjNISLSWHTbWrK2k9bYcdAWRv4I2TWxFpzyhlZu4Q4k+D1fObtmdjjkdH1q0rbXkNuMjEaOlr+dgWWPZRHbfDALybDbis562N4P0KxtaLIZaoDPiYxWdjr0z+u3bPlpjHJ8xtz0Sk7O9Iu14JKROefTtoOu5gW14poMS1Qy3ND3fNtEZb/F65M2xBG21QD0LrdkJjWZ3BK/Hi9e7ur0O6Z8hYbxULzpEkZxjRfZ4OK5r+5rtrhqee6btouWlRuh1jZU9dmet1e3dwb3K81cu8lz7Qg9tBwtta1esudKNDtyzLkbLNgdj+Z3QyZxBeBHutyTOhnLqW5Gj0hJ98sLLze6t0Fgp0QzEGuLjZV4QJrrdRHch7gffGa8muNyzem17K60gDjOiC3DBGrA7pIdmu+RwT9zlaTsdJwz2FXRk1Qi+9+9sa8O9EJWs1ZMnHcc5M3l0TjpnUA/npL+dk87Jk2dnnml3zskd1IPjnMT/c5wtealzTk6SfXZO7s4+p//bDim2AcKtRdmWgqgANZ3NeVEuaM7Z2eesuplTN8/KPNJplSGbzYxHzWYzwzbpB4vKHDqjmtRv6/T092nJz+SNJ51MjqPHc9IHnANnZZ7PpJ/OoB5OT38n70ndoqx16CfKJ4cuZo5zclv62yHFhkrcSLYl+0wnIpFkHTj0u52ccxzNjJ4Om3NyM/WAX3Z2RpBx6pzcTj/FBf+cnHBv9jkBTzrObpYq50c2FFSFc9IXU4U4luYFzpacoJkLrHMyRzgnnaGcBKdwM6leza15vXNyvCAqOHOa+yAo/t8AR7mXrdvDVO1maXbCil2guOnsAOWgJ01nOyTeBgiLaQgEoIn/V0Sb+TwD3tGEZM1iuBnvaDY350W5gpMvfM18BJpx3meLcu7RytSv7Eua+edm8+zsM12TMx43swW9mXPabDbPyjzSHmUeLMp20VEmsTuLFtDumk3q9ZZDW1OHstq0uEn5hnw/k36iH5qp82ZijFOPms7JM9PflD9Nx0mdUl5adMysqM6eQQlOT3+nXJS7Q/ikavqyj/jM7tC3zU6wE+vJWgBynjU6uDjYkRejCcAiTA5PHiP7NOAOY7SH40ya0a12o+0F9nbLThY94bva0fK9orBqTG3LCdHIvyCbNS4tyGZAWRWQFf2bBtxOA26nAbd1wG0NlBXdVouJgFcq5oV4meKOnJCsUYSkbntnQYpXJ+b9xUsTiyxel1j0GC1KLPpQMwBhFRZ2BiwbDxfw5QT4WGcyQsPL09JtTmhL0YmDla2W3XDwmSZ4/fCKvYFCFYl8m3wvRF9hzkmk3Q1U3rclz3jdAD65N5Wt2KuGb7uW7acvQKd8dNFpCVvxQect0n8PbbQub7hBRg7JeczR6cF4oy4aGB1gAgGaevSMpY3QDo4yqa7vLQUG+TIaDzVbXBq47SblS76oZsFt0Vrk0Hfs+Lzb3bEs+TgTkLtVhiNF0LUbvTbZ/Zes/UQJfxAAgg03XDZ8dDlDfAj8IQDD055kx9/iIr6CY6JSKeOS681OlJM8/RhLTSaQ8SLWXQzmckje6bYNF80AGAE+EiXaV16GTVSO9Oeb5ZvN8Hb9sf1Qt+FDHc8qjUHHsyYqE334g8N+UR9g+fv68GTJbjkuWJByILmU49IS0uwaK+VJgaiJyuG+/MLJcdu+0D6g/t6JE6WkqEUoSZYSb6Obj/qJSnx1UjnU3ztxVI6yUByipCZEF1UwI56l+XUsy/ZD9flaHCFm3AnbCbtJbJiVgEL5VY4CS5F+3savchRIyta5fNI2W227Tp2bV5LZMR8lT5++x1cAl+VSzPdTOGK2H6rP1+KkneyPJalbkloW3pbUb9WI6X5iH7P9UH2+Fse+zmDZVnmiMgpo0kUW+AzdawEEWrmWYuX9LArcDXiPozMCKEzfNshcvrPUtvdABM7b8ViDLtxAU/Po22w3Wq8bX04BQrkbuYIhGoo+nkQfhlHfjqdHnz6CvYA+ugZiMqeKLhEx0Rdmcg+WG13jhc5BLIcP5BDfNgPPRRFb6ll4OOr5DXtbjsL3wuRkLTvMY2gFQF6GRgTb87K259q5SBMThQ5ODIM4z3y744WkrOG+qhvuSzRxsKNvOnhrBUuZ5MT+AhHFGi+zK3oeaZFTphKlTfHNkTIxjEznKM3mWcpGuJ6YAEjJjA9OYaYW38c0yNTiohKPVqIvaI1lu7HS9RxyEhA6KrfGBliaAxwnaNWHYaErJOOAde1WdHccmUvBx7CZrQpHHX1jiwrLGI/Eu7G9EoZceDfCY/DtNEMcAt2Ix4uSbzcPctT42yE5bZH3FvTddJSjJ9tUuGlrWta+gpq6K+YwR3n8+ER8UAiy1/s56JHzanktdaEYQ3M3SoG28buFY31SoL+WJQIHch6bvu2a8cl4sM7qdbrZlKSUqHlAgdtFq4mKnHNTlCP/9hXFuDXCaXfLWEm2g9ItGW7h02RjANNMPcfp7oyGKgeDOUXmMU0YanCIDAaZxpmo7IbVaS+jOIRNJDnE7OYRs3tlXgCWErdVdnh6FtyT8ZVcVUVu2N0FaNCVgrsBOd7CDzlA82yQA5x5cdHAszpxPoWege8/i1tbspKSzDNtiWRtr2V0TAcdX09J8ISh7S95gROi3aUeMgsDFBAfLxivC457jNgQ4TMHk1RPRblTkvpzkz0n6ia4SY4jqvTlBp2mdKQvkizb6C8kqQSF5HBfbvCBTDQ6nMyhkqUiufi1cF/UNxIKL+RwPPfEieGG5wbhwclhFpIAmZnhwiMKUuUoHSTqNx7Z5WI6UdmaIXDMdmZEyUGkA1kxtuiuZ+BzyOczuti6DKcJEK2YOY66yJNYTl6/C3Q4ct7ujNxJ7/YoKJLLkfIKt4dPNY4OrwVcEmCisrOoQBezDuTF0T3J6DrGPXkdSnnbdPMRcpIrRbZl5Xh2eXtGhk+CzWeI7ZuBneXsTjfcyEeHetjLUkxUsh7hQVc+ZCin8zIzQKUsGzAcrSsoEWibo2JWbpsjMO48QRPWlGwXLaImpmNr2XTiu2I3UxLcU46R9K7ZOL1Q78T32vHXB2TH91AqtMEf341unzL8TjyUxFeiomNOUWMbX9scL1wby0Do40H8TSNdojZSZEjVSohDecJoLHtoTXuOuzLPgTkSHW1XniMReDTjaxSxIN5eYNguus7HCJcdd8VxW/sYNL4DYC+sRJk4CKvInczdYABW4ww9AunQZyubnB0ZD1DwqZ/9sTBmOT46Ep7cf4t8GynFDkNEUqAz0kwUg/jI0qbjOsHyHlhn29ZMRhNdrtnLfGUpqFJTCbhacVxrojICAPENK4SoAoRvN+HX+nZzonIvQIF2ldh+uPFfKalzwAtI4wCHKm44oMjG7khkRwEialZipA4gttvrwK9GmonKENMN8XMM0KOTO2w/RC/Gw5pb85meG1VU24qvmgeFx/jeRFMv0VTZRGWSj+O5AHIaaTBRKfHbwPFAR0J3HDRUKfE7xkm4j/QFo8UhXvNwCZtcy203y4IcvR1ZczQjUOYzug6CVJ+gVoJGtghdPYs/7ffa9lSJE7I2JbqwAjkoS0EyNRolN2RBcGcTLrlYNVE5BKioSZAkIQ/yuGiGsmMmc0lMjGy9LKOW0FlEkHFLqKbptCd4AElMklnc9xEQWbZyiuQNVFZzIJpswE1LpZyNDBA3JwiJu2qQMclPKEFlATuGy0LUs4USPHJF7BhkmE13w/AYzQXRMdzhqsxwh3WgQU+u+kWT9ROV/RCBAotnh6+mtVHvoXh1281spfYUPcdTnbl0jDXx2p62GWS7JTFQlO2FMLPbtV0rUwTR9FTHcR20s79TR9MrXVxFUHfaDMJqgY0AdIog7ujibrHpuLZPRsQDLBcdszvM0sWvO5QHov0meW4yz1HzxSY6ZcsiHWzclZ5iwt2NcBnv49hAvdd4p0tQCGbqwPY7vTBrLjMAqWvpu+MLzdFarYX+XcX7UtA6rBV7o34TXLZNsjIsm5TpsXNo248bRF3eTsccY3Gn1my3jg9L90dZTAdtZg+6pr8yzkVabafTsX3mu1AJDELfbBt19ruSUjrMQxrdzsxBFtBud4yTXXR8W8dzN/YzMcdt1ZmBbTc7tehb4iCXmRliqVecjmOs1Ev0NWZarNSMZc93rvNcZiijOBrrnTbzNa1uaHhBwPSj5bRMLFytH2AypIa27ZbZ2GCmB6EmWOq4vKPj8VbrRo35Nsu2u4Ftr2CKmTwNb9n27VrHs4GSkLOWWJbxqRO0gCH6cAHJjbzHcoBhOc1murEPPQV3h5j/Soty+/+izydyrSf2zsY7umw/QJdYtp1wiEeYljXN1GczIKO6lqn5rySSmfc++qz432vY8wUpWROblpMigjf/U/ssg3zaoh6Aa8WnrLjhXXL6/0rQMz0GvL7diCfHBgoqdDlwD52lbO8BdEFvKQj9oodoWY2NuiCQI9I52UVrqJUAewG5vY4XGmSMTnTTUTSRg2poHHi8vBrvTbzOjm+W257RodGD59p3o4XRpXFgCi+ZYX8pHIGx+TOXPaPh+WhGPF3ESAvj8aFJ9ncabc838RLnlM5rrmEpwJCbVp9lIwJ3NtpO14jXXEeNn7E+gMXkM5aLJ7E81Ibjr2LnUjo0i5qsB2mbq6bRW8bn6KG5VAtvBQ5qDB6NcHx3tW1Yy37OyRDgBC9wXyZWfQDQk/YhXN8L6PBVDvXafoYquoaYoXXqNZaqWa8NMlSmj79N8NTNeu00rEZBvxP51TFbttGbz+VtoirL2wy4m/KyWa9FCwtxrc8pqKthRrBuOW5RkZM10+/0UCEhpYO4JgfkbHi9sLfaToaYe2idGZi1RDNEadaWnaCLbk9zG9Rqh1S/6q3jLqMfJmr6pUFod+vUS8dondf1TddDUcL7L9C8fej3OgMQE7nfS+ucVtvpJqphWmW7ge16qIxXwXDhfjUVrvG8LgzRV/sVHO3ANvBJAWM8yLeDmmuHXCZ62RSPQfNZbnKFUWCgnQ6HIActG39TsmzyC6X04T64rm+jw4z7QaPgVrhobc1cjbOw2geJpr/wEc04Sev9uDB7gUlOSsYHJfODTtwAZSKLAuUJAyazuNSo4nKM0nW9xood0lkYNjtG29xAK/fIQRLjXByoGhRkm66N7n/DRxrwGZIJR7kMPm8ajz67XmBXWUGL6ChoIyBEpyOdWF0Hm4REt4/Wmb65YlNe788oLatte410GmyU0rpo1aKPvvnXDCorRiDE67gOWLfTsTLZgAWWFHq0nACTNOAtoTMz7HB1Jh79JyJ8mcK5fNghM8C49XWsntk+wudtq2Un7BCXzVrndLgO5kY0VF8Fs7HjdLwknY1OZy9Ll39lx8ulLd3S4J4HWP3wwptpMCh4KoB6X8ZZs0OrDuRUJlVMkYBUzPEyClWT/RwoFymkBUOH5hCo0A3kVCu1GTBj4v4WWE2We+5Gz6S1dDVp+aaLNtoGXdtGPecIqRaRafxNxkvibbobqGtnBoHdWWpvjJW6yCZCq92ZhksSmXdYBWNKdD2uErbJRGmCSWSvm55rTxszoEvLCwPaygzldHUXLY82YX0090G7H4b1NbB+xACVHnTI0ep+z6cNz96MtrXa7iSqHakqzsRVJ9xelHY6m4kQHQhM+otnJoKG2d0WPcQbTFEvl/R3ydpz9Mt2G5EWybZiGVlIHf26Eq0oIsujDtPrjXy7Za/nlhuhjjGZurFP9ZxVs40WBJOzio6n3iDyipvlK/YpXvyV9fDCm+Nhzo+zMn2SbQ3TbaDRl4cur27gYe/WSGY2yKaNcD04C88srDnhMnrcgZ/CZRutiEFLC4zQDFb2YGnklijR8X+NcH0g0pDVg1nd9obZxutLyQwXuXvkzo3aWpRhQdvLz6wkurJhSgbcRYpTaLbQXYHpMuTN8Te4rh06KPpbonLnuPi4YscNzyKS6MP1bvJkkaYYHXblrRronpVdGUV0MpgZrOyM5HjTTS26+MLzt2bFKD92RCcExNP0JITjaFkYSQKc0eiypUZ0DaNhrPl4hD2WQsmy3POOG0ZnKU2C81KGLjdr+LCtkLr4MbBNP34DPrUQT35h127bW+6YrhstQzeXHDSVWq0ZU6gTF5pOO1rV5x9PUeJvgG6gwkWctBFmO4fM4jfgxhQYy67YGwY6RN33cxd0Vvt1hq+mne61+38RXv05OTxTraEXnVvmLHfzKJ/PRKTdH59dCzpZykfhr/faJw5gGF+xSW7XJLe65rzcx6DwMtXJrDKpd+RW4lzk+4XbNwV2bgrc6AOGEpQNZ+51HS+B8SWvE3wovfF1lA+iBcslLyQrlQ/woehQkTE+hXNbQyfBzk7vWTId1InteHbdyLQdh81e6MV7OcjHVpdYHlQAoxUoRL+PRvE0tm8bq6bvmG4YDNDKyErHs9njtK7IkacJFmS6ZnvjOjtZHrc3o8RL8GLVDOiOHNWAj0FspF+ZohMOg4nKdB+uokn51NEt+nBk+q0evquJrDhI3M7dFLeBjfpLIbrNGU4Whjt8+kTyymp/ToPUxSTHRXQJVM9tRJl3lAPTj+TT+vGbQifz/9Hi4vNvjmPczJGbues3w30/76RegWcZ2m07iu5EBXxnskEP3TWfl01UFkrcxBUwXm8brb+1JirzfbpMdwgGDQ/dYg+Waa5DvK3WbdgTlXMht9GXKiqCkYSRJAkfLZbCB36bbdQ96nZtCxfJ+Jwl3KVBxgmdOmqbQZjr6QyZ3a5hr6M9j3jEeH3meaKyzWyG+GMa/sQR9JpNZ327QR1VGbbJh+RdGSE+/QXL9wFy3CFGNwZCjux1J9wLyBu+GSzbVtaJ3en65FLkPUbh/EzfDkLPt3cVNTj6I7Tc76F6Y2CD58TvzPhJ1mh2TKe95K1vNYw1M+hESynw9qMtkYgMAEJ05bdhBKHleORMrTPjJzSsTB7QaxIOn3e1yzC6UZQDu91MOpCaYXQ6Zne3kY2OhW7/Q17WDaOxvm4uOatVlPGrHScaXpHrTfAXMdLZNZZsNHCwgvAmuDGXPLShLghP9OFm2QzQ1Y+dJafV83qB0e0ttZ0GPm+9lnUe9BHK/p0kgTyv3AknjIezrvEm3eSeGPJpEd10gIYmx7Jos+daJmpYzDaIn5vFS6PeJ5/Ee7GE50R6osRpEodc6uD7aGzU5uMz3/HAuRmNfydgFNf/DLgPg0ar7S2ZbbIY0rDQppKFmYUppnJ+am6GqZybXaizlbXZGltZnWV7Ozs3N8tWzs5zlDML02zl9NQCW1mfnuMo6xzlFMfbmYU6O7Qz87Nsb2fmZqsc5TQ7y6bnq2yX03OczK7PL1QHmMo5tq/12TmOr9Wpeaayxkv36sLU/H6mcnaBk3zV2Zk6x2ltrsrWTk/Vp9ja+vwcu4xV6/Oc19bn5msc7ez8LFtbm6/PcbT1GU5K1Wq1aZ62ynVb5bqd4oWqusBzW+XGt8pLq1p1jhfm6hz3vbNcn2e4MZrhFI1adZrrc50b3yrX7RQvRlPctJriptXUHKfE1qZmORWlNlXnhqrGDRW31E1xS93UFCdU1YUFTg5WFxY48a0uzHN9nl3gajnxrS7wamh1YZpTNqoLvHJVXeCZq+pCbYar5caXY7mRluN0foEXofkF3mvn53kZOD/H1/KSan6WG+YZbphnuD7Xedk7X+e6rXHdVrmh4hmG6hw3j+Z4BrjKbayqM/O8YjXNM/zIMUc7xasKU/M1ThWcmuVreWZjarrOc1ubnh1kaI8toD50fYipXjBm5uem97L1bKfzxsx8rc52Os9WzbJV0+wX1lFYa2ynnLDU9jBVTEfzs8ygzM8YC7P16jBHX52tzU6x/Z5hu62jIlxnJ9E8O57z7NSZZ3afj81PzXB07KRjZ/AcO1Xn5tiqGWaCz9WNhdnaDNspO03m2Nk0N4WyaYad1HNTzMjPMcM6i/q8sxz9AooLO8Cz7Do3O2/U6rPs8M7Os53Oobeyy8csO194L2RnySy7Js/i3GTXjllO4nBiUGVmFrsmzyDbW2dnFtZPsQvmDMrMKrsmzCwwS9/MPBo61ThuOSpOiGeNepUXoll2iLDZmmGn/gwnIabRa+fYfk+j2M6wDcPMNNttHbtll1BekGtsb6vYW47bKlvFtgzsNJrGhYXdGk8voOrNtlfTCyjA0+w0nGaXtmlc2qbZaTjNth3TyHZU2Qk5PYf9ZtuCaXa5mJ7FbuvMFmiaY36m2WV5egb7y0lMBNQ5nYZpdhs9jQtznZMRbOM3jYxflW38puvs19ZQ/ZznpHON/dqqsTA7xQlxle10Ck2ncor1lDEzN8eJ8RQyDezqPY2b4XlOvKdwVrFN//QU8+V1bODZ6YL1Uxw9qrNTbHNaZ1e5+jyOF7vjVef0ZuvsKlOfxdOEC2x/ZznhZdemOrvA16fxK9nWo84eQ9TZBq9eM2ana1Ps1+ICP8cu8HW2UapXsVt2/a6zrXudbd3ZkanhksI2ZDX2DPax2gLbINTYGV2b5+nmObo5jo7dW6qhhoBTWWpzxux0lRPNOU7yzNVG2LqqUa3OTnFiO8d7Kydx2R2p2iz60sGuE7VZNG3Oi+0sJ7yznNzh9KprnMxBg9IpTnBm2NamxrbutWn0qYht5WrTyILO1Njv5XRYauxWo4bayakptrf1aY6Ok+OcGlxFsyXs4VetipKC3U+qVdF3Kk4qs3sj1QXcxLGDXWUHqzqPgsVuL6qo61efn+cCU7Oc/luV43YOu2Wb2Sq7hlVnUHqz+3DVGfxNi93qVNkTF9U628RV67McHbtYVets81XlNCdVdtemWsNf1zhh5QwQq+yixjFLVY4ZrXKmgqqcqaDqLoZmK5KbRreHlkE6ftgz26NE5LgN38ar8JIFR+h8KHx01S6CoMW9qbIb+ucQebIYYZthNKILA3ro3Buj7TUoWdtbI7KdsQzf3byE7tVFS95HrrrNFVeS63ovIfdAZAUTlUqeuMppty/yve5lZhBevE7W5twJndeHSXLPbwm5+7LLLjfucnvjksuvuGxxkXo4G/2+9K5EduldBUEUJEEWFEEVNEEXNgmnCacLm4QzhE3CmcIm4SzhbOEcYbOwRdgqbBK2CduFHcJOYZewW9gj7BUGhH3CfmFQ2CQMCcPCJmFEGBXGhE3CuHBAOCgcEiaEinBYOCJMCkeFY8K5wpRQFWpCXZgWZoRZYU6YFxaEReEWwibhuLBJOE84IWwSzhduKdxKuEDYJFwoXCRcLGwSbi3cRritsEnYJFwibBJuJ1wqXCZcLtxeuINwhXBH4U7CnYW7CFcKdxWuEu4m3F3YJNwDs/cUrhauEa4VDPy0SbiXYEa/0N+S0BA2CZZgC0383BI2CcuCI5wUVoS20BE2Ca7gCV3hVMT7QiCEQk/YJKwKa8ImYV3YEK4T7i3cR7ivcD0m7ifcH//7AOGBwibhQcKDhU3CQ7DkocLDhIcLjxAeKTwKPz9aeIywSXis8Djh8cIThCcKTxI2CU8WniJsEp4qPE3YJDxdeIbwTOFZwrOF5wjPTUL8POH5wguEFwov+n/S98XCSyLpS4WXCS8XXiG8Mnp+lfBq4TXCa4XXCa8X3hDJ3ii8SXiz8BbhrcLbhLcL7xDeKbxLeLfwHuG9VHq8T3i/8AHhBkpyUPig8CHhw8JHhI8KH4tkHxc+IXxS+JTwaeEzwmeFTcLnhM8LXxC+KHxJ+LLwFeGrwteErwvfEL4pfEvYJHxb+I7wXeF7wveFHwg/FH5E+Rz//Vj4ifBT4WfCz4VfCL8UfkVpfi38Rvit8Dvh98IfhD8KfxL+LPxF+KvwN+Hvwj8yPvxT+Jfwb+E/wo2CIIqiJN54Y6qTRUVURU3UxU3iaeLp4hnimeJZYqw9WzxH3CxuEbeK28RBYbu4Q9wh7oy0u8Td4h5xrzggLgv7xP3ioFgMO/obEofFEXFUHBPHsftl4YB4UDwkTogV8XDBzRFxUjwqHhPPFafEqlgT6+K0OCPOinPivLggLoq3EI+L54knxBuE88VbirdivPMC8ULxIvFi8dbi9cJtxNuKl4i3Ey8VLxMvF28v3kG8Qjwh3lG8E8PtncW7iFeKdxWvEu8m3l28h3hP8WrxGvFa0RDvJZriktgQLdEWm2JLXBYd8SQOy4rYZvi2SeiIruiJXbErnhJ9MRBDsSeuimviurghXifeW7yPeF/xevF+4v3FB4gPFB8kPlhcFB8iPlR8mHhCfDgzhuTvEeIjxUeJjxYfIz5WfJz4ePEJ4hPFJ4lPFp8iPlV8mvh08RniM8Vnic8WnyM+V3ye+HzxBeILxReJLxZfIr5UfJn4cvEV4ivFV4mvFl8tvkZ8rfg68fXiG8Q3im/ivpP992bxLeJbxbeJbxffIb5TfJf4bvE94nvF94nvFz8g3iB+UPyQ+GHxI+JHxY+JHxc/IX5S/JT4afEz4mfFz4mfF78gfhGXji+JXxa/In5V/Jr4dfEb4jfFbzHD8m3xO+J3xe+J3xd/IP5Q/JH4Y/En4k/Fn4k/F38h/lL8lfhr8Tfib8Xfib8X/yD+UfyT+GfxL+Jfxb+Jfxf/If5T/Jf4b/E/4o2iIImSJMmSIqmSJunSJuk06XTphHi+eIZ0pnSDcJZ0tnTzUqO/v3OkzdIWaau0Tdou7ZB2SrJwRNwl7ZJ2S3ukvdKAtE/aLw1KQ9KwNCKNSmPSuHRAOigdkiakinRYOiJNSkelY9K50pRUlWpSXZqWZqRZaU6alxakRekW0nHpPOmrwgnpfOmW0q2kC6QLpYuki6VbS7eRbitdIt1OulS6TLpcur10B+kK6Y7SnaQ7S3eRrpTuKl0l3U26u3QP6YR4QryndLV0jfQG8VrJkO71fzU10J8pLUkNyZJsyZaaUktalhzppLQitaWO5Eqe1JVOSb40KQZSKPWkVWlNWpc2pOuke0v3ke4rXS/dT7q/9ADpgdKDpAdLD5EeKj1Merj0COmR0qOkR0uPkR4rPU56vPQE6YnSk6QnS0+Rnio9TXq69AzpmdKzpGdLz5GeKz1Per70AumF0oukF0svkV4qvUx6ufQK6ZXSq6RXS6+RXiu9Tnq99AbpjdKbpDdLF0lvkd4qvU16u/QO6Z3Su6R3S++R3iu9T3q/9AHpBumD0oekD0sfkW688aPS3YWviR+TPi59Qvqk9Cnp09JnpM9KNwifkz4vfUH6ovQl6cvSV6Sv/l9P2//tv69JX5e+IX1T+pb0bek70nel70nfl34g/VD6kfRj6SfST6WfST+XfiH9UvqV9GvpN9Jvpd9Jv5f+IP1R+pP0Z+kv0l+lv0l/l/4hTYr/lP4l/Vv6j3SjJMiiLMmyrMiqrMm6vEk+TT5dPkM+Uz5LPls+R94sb5G3ytvk7fIOeae8S94t75H3ygPyPnm/PCgPycPyiDwqj8nj8gH5oHxInpAr8mH5iDwpH5WPyefKU3JVrsl1eVqekWflOXleXpAX5VvIx+Xz5BPy+fIt5VvJF8gXym+VLpIHhYvlW8u3kW8rXyLfTr5Uvky+XH6AcHv5DvIV8h3lO8l3lu8iXynfVb5Kvpt8d3m3cA/5nvLV8tXyNfKZ0rWyId9LNuUluSFbsi035Za8LDvySXlFbssd2ZU9uSufklF9RH/Xyr7sy68QAxlZ8FAO5Wvlnoz+PV+8p3QGrrHXCKvyZ6U1+QZhXd6Qr5PvLd9Hvq98vXw/+f7yA+T/7bLw/87fA+UHyQ+WHyI/VH6Y/HD5EfIj5UfJj5YfIz9Wfpz8ePkJ8hPlJ8lPlp8iP1V+mvx0+RnyM+Vnyc+WnyM/V36e/Hz5BfIL5RfJL5ZfIr9Ufpn8cvkV8ivlV8mvll8jv1Z+nfx6+Q3yG+U3ym+S3yS/WT4ovEV+q/w2+e3yO+R3yu+S3y2/R36v/D75/fIH5BvkD8ofkj8sf0T+qPwx+ePyJ+RPyp+SPy1/Rv6s/Dn58/IX5Enxi/KX5C/L/5a+In9V/pr8dfkb8jflb8nflr8jf1f+nvx9+QfyD+UfyT+WfyL/VP6Z/HP5F/Iv5V/Jv5Z/I/9W/p38e/kP8h/lP8l/lv8i/1X+m/x3+R/yP+R/yv+S/y3/R75RFhRRkRRZURRV0RRd2aScppyunKGcqZylnK2co2xWtihblW3KdmWHslPZpexW9ih7lQFln7JfGVSGlGFlRBlVxpRx5YByUDmkTCgV5bByRJlUjirHlHOVKaWq1JS6Mq3MKLPKnDKvLCiLyi2U48p5ygnlfOWWyq2UC5QLlYuUi5VbK7dRbqt8VbhEuZ1yqXKZcrlye+UOyhXKHZU7KXdW7qJcqdxVuUq5m3J35R7KPZWrla5wjXKtYij3UkxlSWkolmIrTaWlLCuOclJZUdpKR3EVT+kqpxRfCZRQ6Smrypqyrmwo1yn3Vu6j3Fd5j3S9cj/l/soDlAcqD1IerDxEeajyMOXhyiOURyqPUh6tPEZ5rPI45fHKE5QnKk9SdgtPVp6iPFV5mvJ05RnKM5S/CM9UnqU8W3mO8lzlecrzlRcoL1RepLxYeYnyUuVlysuVVyivVF6lvFp5jfJa5XXK65U3KG25Lb9ReZNy44033vhm5S3KW5W3KW9X3qG8U3mX8m7lPcp7lfcp71fer3xAuUH5oPIh5cPKR5SPKh9TPq58Qvmk8inl08pnlM8qn1M+r3xe+YLyReVLypeVryhfVb6mfF35hvIN5ZvKt5RvKyfE7yjfUVBf5Wrpu8r3qL9XiN9XAvkHyg+VV4pN8XyxJ79K/J6CbMaPlB8rP1Z+ovxUCWXy9wrxU9KZ0s+U7yufloibz0jE1avEnyu/UH6p/Eq5Wvq18mvlN8pvlN8qv1N+r/xB+aPyJ+XPyl+Uvyp/U/6u/EP5p/Iv5d/Kf5QbFUEVVUmVVUVVVU3V1U3qaerp6hnqmer/tjX4///+v/p3lnq2eo66Wd2iblW3qdvVHepOdZe6W92j7lUH1H3q28X96qA6pA6rI+qo+nRhTB1XD6gH1UPqhFpRK+ph9Yg6qR5Vj6nnqlNqVa2pdXVanVFn1Tl1Xl1QF9VbqMfV89QT6vnqLdVbqReoF6oXqRert1Zvo95WvUS9nXqpepl6uXp79Q7qFeod1Tupd1bvol6p3lW9Sr2benf1Huo91avVa9Rr1atVQ72XaqpL6rvFhupIlmqrTbWlLquOelJdUdtqR3VVT+2qp1RfDdRQ7amr6pq6rm6o16n3Vu+j3le9Xr2fen/1AeoD1QepD1Yfoj5UfZj6cPUR6iPVR6mPVh+jPlZ9nPp49QnqE9UnqU9Wn6I+VX2a+nT1Geoz1Wepz1afoz5XfZ76fPUF6gvVF6kvVv8uvUR9qfoy9eXqK9QPiK9UX6W+Wn2N+lr1derr1Teok+Kk+EY1kN6kvll9i/pW9W3q29V3qD9Q3qm+S323+h71ver71PerH1BvUD+ofkj9sPoR9aPqx9SPq59QP6lK8qfUT6mfVj+jflb9nPp59QvqF9UvqV9Wv6J+Vf2a+nX16+o31G+q31K/rX5H/a76PfX76g/UH6o/Un+s/kT9qfpT9WeqIv9c/YX6S/VX6q/V36i/VX+n/l79g/pH9U/qn9W/qH9R/6r+Tf27+g/1n+q/1H+r/1FvVAVN1CRN1hRN1TRN1zZpp2mna2doZ2pnaWdr52ibtS3aVm2btl3boe3Udmm7tT3aXm1A26ft1wa1IW1YG9FGtTFtXDugHdQOaRNaRTusHdEmtaPaMe1cbUqrajWtrk1rM9qsNqfNawvaonYL7bh2nnZCO1+7pXYr7QLtQu0i7WLt1tpttNtql2i30y7VLtMu126v3UG7QrujdiftztpdtCu1u2pXaXfT7q7dQ7undrV2jXatZmj30kxtSWtolmZrTa2lLWuOdlJb0dpaR7tWdjVP62qnNF8LtFDraavamraubWgb2nXavbX7aPfVrtfup91fe4D2QO1B2oO1h2gP1R6mPVx7hPZI7VHao7XHaI/VHqc9XnuC9kTtSdqTtadoT9Wepj1de4b2TO1Z2rO152jP1Z6nPV97gfZC7UXai7WXaC/VXqa9XHuF9krtVdqrtddor9Vep71ee4P2Ru1N2pu1t2hv1d6mvV17h/ZO7V3au7X3aO/V3qe9X/uAdoP2Qe1D2oe1j2gf1T6mfVz7hPZJ7VPap7XPaJ/VPqd9XvuC9kXtS9qXta9oX9W+pn1d+4b2Te1b2re172jf1b6nfV/7gfZD7Ufaj7WfaD/Vfqb9XPuF9kvtV9qvtfPF32i/1X6n/V77g/ZH7RrlAukCCc0vdIWrlQsk9P+u8EbpjdIF0p+0P2t/0f6q/U37u/YP7Z/av7R/a//RbtRu1ARd1CVd1hVd1TVd1zfpp+mn62foZ+pn6Wfr5+ib9S36Vn2bvl3foe/Ud+m79bdJF8t79L36gL5P368P6pdIQ/opZVgf0Uf1MX1cP6CvKwf1Q/qEXtEP66ZwRJ/Uj+rH9Pso5+pTelWv6XV9Wp/RZ/U5fV5f0Bf1Byi30I/r5+kn9PP1W+q30i/QL9Av1C/SL9Zvrd9Gv61+iX47/VL9Mv2OMvm7XL+9fgf9Cv2O+p30O+t30a/U76pfpd9Nv7t+D/2e+tX6Nfq1uqHfSzf1Jb2hW7ql29Qf6gMU23/Ua7D1pt7UW3pLP0dY1h19u3BSX9Hbekd3dU/v6qd0Xw/0UO/pq/qavq5v6Nfp99bvo99Xv16/n35//QH6A/UH6Q/WH6I/VH+Y/nD9Efoj9Ufpj9Yfoz9Wf5z+eP0J+hP1J+mov/Rk/Sk66i89VX+a/nT9Gfoz9Wfpz9afoz9Xf57+fP0F+gv1F+kv1l+iv1R/mf5y/RX6K/VX6a/WX6O/Vn+d/nr9Dfob9Tfpb9bfor9Vf5v+dv0d+jv1d+nv1t+jv1d/n/5+/QP6DfoH9Q/pH9Y/on9U/5j+du3Tysf1T+if1D+lf1r/jP5Z/XP65/Uv/J/27SlYkyZaE/DWtytzpVVt27Zt273btm3btm3btm3b7one/39OnDmD25mLrifeFVlVmXVbWRG1vPNe9ZAL3kXvdNgl75J32bviXf3XNW918HXvhnfTu+Xd9u54d7173n3vzw7tgffQe+Q99h57T7yn/3rmPfOee8+9F94L7+V/88p75b32XntvvDfe2//iXaT33gfvo/fJ++x99tqEfvH+7Pe+el+962H/e/98EdYIqRXyzfvu/fB+er+8316n0E6hQSgYXQ2EoBAUGinsv0gaHEABFI48hBBGGAEiiCCKKGKIIY44EkggiSRSSCODLHLIR1FQFBQVRUPRUQwUE8VCsVEcFBfFQ/FRApQQJUKJURKUFCVDyVEKlBKlQqlRGpQWpUPpUQaUEWVCmVEWlBVlQ9lRDpQT5UK5UR6UF+VD+VEBVBAVQoVREeQFiqJiqHikEqgkgkApVBqVQWVROVQeVUAVUSVUGVVB/6/3FH/99ddff/31119//fXXX3/99ddff/31119//f+lKqqGqqMaqCaqhWqjOihKoC6KQPVQfdQANUSNUGPUBDVFzdCW4OaoBWqJWqFWqDVqg9qitqgdao/aow6oI+qEOqMuqAvqirqhbqg76oF6oJ6oF+qFeqM+qC/qi/qh/qg/GoAGooFoEBqMBqMhKE5gKBqGhqN4gRFoJBqFRqMxaCwah8ajCWgimoQmoyloKpqGpqPpaAaaiWah2WgOmovmofloAVqAFqJFaDFagpaiZWg5WoFWolVoNVqD1qJ1aD3agDaiTWgzShnYgraibWg72oF2ol1oN9qD9qJ9aD86gA6iQ+gwOoyOoKPoGDqOTqCT6BQ6jc6gM+gsOofOowvoQchFdAldRlfQVXQNXUc30E10C91Gd9BddBfdQ/fRA/QQPUKP0WP0BD1Fz9Bz9AK9RK/Qa/QGvUXv0Hv0AX1En9Bn9AV9Rd/Qd/QD/US/0G8UhCeHBuMQHIrDcACHYw8jjDFggilmmON8AYElVlhhjQ222GEfR8FRcTQcHcfAMXEsHBvHwXFxPBwfJ8AJcSKcGCfBSXEynBynwClxKpwap8ElAmlxOpweZ8AZcSacGWfBWXE2nB3nwDlxLpwb58F5cT5cNpAfF8AFcSFcGBfBRXExXByXwCVxKVwal8FlcTlcDpfH5XEFXBFXwpVxZVwFV8XV/q3VcQ1cE9fCtXFtXAfXxRG4Hq6PG/xP44Z4fWjdQCPcGDfBTSM1w81xC9wSt8KtcRvcFrfD7XEH3BF3wp0jdcFdcTfcHffAPXEv3PtffXBf3A/3xwPwQDwID8ZD8FA8DA/HI/BIPAqPxmPwWDwOj8cT8EQ8CU/GU/BUHBE0DU/HM/BMvDl0Fp6N5+C5eB6ejxfghXgRXoyX4KV4GV6OV+CVeBVejdfgtXgdXo834I14E96Mt+CteBvejnfgnXgX3o334L14H96PD+CD+BA+jI/go/gYPo5P4JP4FD6Nz+Cz+Bw+jy/gi/gSvoyv4Kv4Gr6Ob+Cb+Ba+je/gu/gevofv4wf4IX6EH+Mn+Cl+hp/jF/glfoVf4zf4LX6H3+MP+CP+iD/hz7hb4Av+ir/h7/gH7h74iXsEfuHfOAiCIRhCIBRCIQwCEIBw8AABAgwABCgwYMBBgAQJCjRoMGDBgQ8+RIGoEA2iQXSIAb0DMSEWxIY4EBfiQXxIAAkhESSGJJAUkkFySAEpIRWkhjSQFtJBesgAGSETZIYskBWyQXbIATkhF+SGPJAX8kF+KAAFoRAUhiJQFIpBcSgBJaEUlIYyUBbKQXmoABWhElSGKlAVqkF1qAE1oRbUhjpQFyKgHtSHBtAQGkFjaAJNoRk0hxbQElpBa2gDbaEdtIcO0BE6QifoDF2gK3SFbrAg0B16QKegntATekFv6AN9oR/0h/4wAAbCwsAgGAxDYCgMg+EwAkbASBgFo2E0jIGxMA7GwwSYABNhEkyGKTAVpsF0mAEzYSbMgtkwB+bCPJgH82EBLISFsAgWwxJYAkthGSyH5bACVsIqWA1rYC2sg/WwATbCJtgMW2ALbIVtsB22ww7YCbtgN+yBPbAX9sF+OBd6AA7CITgMR+AIHIVjsDhwHE7ACTgJp+A0nIYzcBbOwTk4DxfgIlyES3AZrsBVSBZ0Da7DDbgJSwOB4FtwG+7AXbgH9+EBPIRHsCrwGJ7AU3gGz+EFvIRX8BrewFt4BymD38MH+Aif4DN8ga/wDb7Dd/gBP+EX/IYgEkxCSCgJIwESTjyCCCZACKGEEU4EkSReiCKaGGKJIz6JQqKSaCQ6iUFiklgkNolD4pJ4JD5JQBKSRCQxSUKSkmQkOUlBUpJUJDVJQ9KSdCQ9yUAykkwkM8lCspJsJDvJQXKSXCQ3yUPyknwkPylACpJCpDApQoqSYqQ4KUFKklKkNClDypJypDypQCqSSqQyqUKqkmqkOqlBapJapDapQ+qSCFKP1CcNSEPSiDQmTUhT0ow0Jy1IS9KKtCZtSFvSjrQnHUhH0ol0Jl1IV9KNdCc9SE/Si/QmfUhf0o/0JwPIQDKIDCZDyFAyjAwnI8hIMoqMJmPIWDKOjCcTyEQyiUwmU8hUMo1MJzPITDKLzCZzyFwyj8wnC8hCsogsJkvIUrKMLCcryEqyiqwma8haso6sJxvIRrKJbCZbyFayjWwnO8hOsovsJnvIXrKP7CcHyEFyiBwmR8hRcowcJyfISXKKnCZnyFlyjpwnF8hFcolcJlfIVXKNXCc3yE1yi9wmd8hdco/cJw/IQ/KIPCZPyFPyjDwnL8hL8oq8Jm/IW/KOvCcfyEfyiXwmX8hX8o18Jz/IT/KL/Ca/SRANpiE0lIbRAA2nHkUUU6CEUsoop4JKqqimhlrqqE+j0Kg0Go1OY9CYNBaNTePQkNC4NB6NTxPQhDQRTUyT0KQ0GU1OU9CUNBVNTdPQtDQdTU8z0Iw0E81Ms9Cs9FQgG81Oc9CcNBfNTfPQvDQfzU8L0IK0EC1Mi9CitBgtTkvQkrQULU3L0LK0HC1PK9CKtBKtTKvQqrQarU5r0Jq0Fq1N69C6NILWo/VpA9qQNqKNaRPalDajzWkL2pK2oq1pG9qWtqPtaQfakXainWkX2pV2o91pD9qT9qK9aR/al/aj/ekAOpAOooPpEDqUDqPD6Qg6ko6io+kYOpaOo+PpBDqRTqKT6RQ6lU6j0+kMOpPOorPpHDqXzqPz6QK6kC6ii+kSupQuo8vpCrqSrqKr6Rq6lq6j6+kGupFuopvpFrqVbqPb6Q66k+6iu+keupfuo/vpAXqQHqKH6RF6lB6jx+kJepKeoqfpGXqWnqPn6QV6kV6il+kVepVeo9fpDXqT3qK36R16l96j9+kD+pA+oo/pE/qUPqPP6Qv6kr6ir+kb+pa+o+/pB/qRfqKf6Rf6lX6l3+h3+oP+pL/obxrEglkIC2VhLMDCmccQwwwYYZQxxplgkimmmWGWOeazKCwqi8aisxgsJovFYrOLgTgsLovH4rMELCFLxBKzJCwpS8aSsxQsJUvFUrM0LC1Lx9KzDCwjy8QysywsK8vGsrMcLCfLxXKzPCwvy8fyswKsICvECrMirCgrxoqzEqwkK8VKszKsLCvHyrMKrCKrxCqzKqwqq8aqsxqsJqvFarM6rC6LYPVYfdaANWSNWCPWmDVhTVkz1py1YC1ZK9aatWFtWTvWnnVgHVkn1pl1YV1ZN9ad9WA9WS/Wm/VhfVk/1p8NYAPZIDaYDWFD2TA2nI1gI9koNpqNYWPZODaeTWAT2SQ2mU1hU9k0Np3NYDPZLDabzWFz2Tw2ny1gC9kitpgtYUvZMracrWAr2Sq2mq1ha9k6tp5tYBvZJraZbWFb2Ta2ne1g1wI72S62m+1he9k+tp8dYAfZIXaYHWFH2TF2nJ1gJ9kpdpqdYWfZOXaeXWAX2SV2mV1hV9k1dp3dYDfZLXab3WZ32F12j91nD9hD9og9Zk/YU/aMPWcv2Ev2ir1mb9hb9o69Zx/YR/aJfWZf2Ff2jX1nP9hP9ov9ZkE8mIfwUB7GAzycPw71OOKYAyeccsY5F1xyxTU33HLHfR6FR+XReHQeg8fksXhsHofH5fF4fJ6AJ+SJeGKehCflyXhynoKn5Kl4ap6Gp+XpeHqegWfkmXhmnoVn5dl4dp6D5+S5eG6eh+fl+Xh+XoAX5IV4YV6EF+XFeHFegpfkpXhpXoaX5eV4eV6BV+SVeGVehVfl1Xh1XoPX5LV4bV6H1+URvB6vzxvwhrwRb8yb8Ka8GW/OW/CWvBVvzdvwtrwdb8878I68E+/Mu/CuvBvvznvwnrwX78378L68H+/PB/CBfBAfzIfwoXwYH85H8JF8FB/Nx/CxfBwfzyfwCXwin8Qn8yl8Kp/Gp/MZfCafxWfzOXwun8fn8wV8IV/EF/MlfClfxpfzFXwlX8VX8dV8DV/L1/H1fAPfyDfxzXwL38q38e18B9/Jd/HdfA/fy/fx/fwAP8gP8cP8CD/Kj/Hj/AQ/yU/x0/wMP8vP8fP8Ar/IL/HL/Aq/yq/x6/wGv8lv8dv8Dr/L7/H7/AF/yB/xx/wJf8qf8Wc8LPQ5f8Ff8lf8NX/D3/J3/D3/wD/yT/wz/8K/8m/8O//Bf/Jf/DcPEsEiRISKMBEQ4cITSGABgggqmOBCCCmU0MIIK5xwwhdRRFQRTUQXMURMEUvEFnFEXBFPxBcJREKRSCQWSURSkUwkFylESpFKpBZpRFqRTqQXGURGkUlkFllEVpFNZBc5RE6RS+QWeURekU/kFwVEQVFIFBZFRBFRVBQTxUUJUVKUEqVFGVFWlBPlRQVRUYSH/lFJVBZVRFVRVVQT1UUNUVPUErVFHVFXRIgXgXqivmggGopGorFoIpqKZqK5aCFailaitWgj2op2or3oIDqKjqJTpM6is+giuopuort4FeghekbqJXqLD4E+oq/oJ/qLAWKgGCQGiyFiqBgmhosRYqSA0H/+wR0lRosxYqwYJ8aLCWKimCQmiyliqpgmpovpYoaYGWmWmC3miDlirpgn5osFYqFYJBaLJWKpWCaWixVipVglVos1Yq1YK9ZFWi/Wi7DwDWKj2CQ2i0D4FrE10jaxXewQO8UusVuEh+8ReyPtE/vFAXFQHBKHxRFxVHjhx8RxcUKcFKfEaXFGnBXnxHlxQVwUl8RlcVlciXRVXBXXxHVxQ9wUt8RtcVvciXRX3BX3xH3xQDwUj8Rj8UQ8Fc/Ec/FCvBSvxGvxRrwV78R7gcM/iKrBH8Un8Vl8EV/FN/FdQPgP8VP8Er9FkAyWP0JDZKgMk2EyIMOlJ5HEEiSROpRKJrkUUkgpldTSSCud9GUUGVVGldFkNBldxpAxZSypw2PL2DKOjCvjyngyvnThCWRCmUgmlklkUplMJpcpZEqZSqaWaWRamU6mlxlkRplJZpZZZFaZTWaXOWROmUvmkrllHplX5pP5ZQEZCCsoC8nCsogsKovJ4rKELClLyUPBpWUZWVaWk+VlBVlRVpKVZRVZVVaT1WUNWVPWkrVlHVlXRsh6sr5sIBvKhrKRbCybyKYyd3gz2Vy2kC1lK9latpFtZTvZXnaQHWUn2Ul2ll1kF9lVdpPdZQ/ZU/aSvWUf2Vf2k/3lADlQDpIFwguED5ZD5FA5TA6XI+RIOUqOlmPkWDlWjpPj5QQ5UU6Sk+RkOUVOldOkCZsuZ8iZcta/Zss5cq6cJ+fLBXKhXCQXycVyiVwql8nlcoVcKVfJVXK1XC3XyLVynVwvN8iNcqPcJDfLLXKr3Ca3yx1yp9wpd8ndco/cK/fJ/fKAPCi3BB2ShyMdkUflMXlcHpcn5El5Sp6Wp+UZeVaek+flBXlBXpSX5GV5RV6RV+U1eV3ekDflTXlL3pZ35F15T96XD+QD+VA+ko/lE/lUPpXP5HP5Qr6QL+Ur+Vq+kW/kW/lOvpcf5Af5UX6Sn+UX+VV+k9/kd/lD/pA/ZenwX/K3DFLBKkSFqjAVUOHKU0hhBYooqpjiSiiplNJKK6OscspXUVRUFVVFU9FVDBVTxVSxVGwVR8VVEUERQfFUfJVAJVSJVCKVWCVRSVUylVylUClUSpVKpVZpVBqVVqVT6VUGlVFlVJlUZpVFVQvPqrKp7Cp2WA6VU+VSuVRulUflVflUPpVfFVAFVSFVWBVRRdX4kGKqmCquSqiSqpQqpUqrMqqMKqvKqfKqgqqgKqpKqrKqoqqqaqq6qqFqqpqqlqqt6kSqq+qqCFVPRQTVVw1UQ9VINVZNVFPVNvyPZqq5aqFaqlaqtWqj2qp2qr3qoDqqTqqz6qK6qm6qu+qhuob3VD1VL9VL9Va9VR/VR/VVfVU/1U/1V/3VADVADVQD1SA1SA1Wg9UQNUQN/T8apoarEWqkGqVGqzFqrBqnxqsJaqKapCarKWqqmqamqxlqppqlZqs5aq6ap+arBWqhWqQWqyVqiVqqlqnlaoVaqVap1WqNWqvWqnVqnVqvNqiNapParLaorWqb2q52qJ1ql9qt9qi9ap/arw6og+qQOqyOqKPqmDquTqiT6pQ6rc6os+qcOq8uqIvqkrqsrqir6pq6rm6om+qWuq3uqLuqM+vK7qn76oF6qB6px+qJeqqeqefqhXqpXqnX6o16q96p9+qD+qg+qc/qi/qqvqnv6of6qX6p3ypIB+sQHarDdECHa08jjf8FmmiqmeZaaKmV1tpoq50+4vk6io6qo+noOoaOqWPp2DqOjqvj6fg6gU6oE+nEOolOqpPp5DqFTqmThabSqXUanVan0+l1Bp1RZ9KZdRadVWfT2XXRsD9y6MtB/zc5dS6dW+fReXU+nV8X0AV1IV1YF9FFdFFdTBfXJXRJXUqX1mV0WV1Ol9cVdEVdSVfWVXRVXU1X1zV0TV1L19Z1dF0doevp+rqBbqgb6ca6iW6qm+nmuoVuqVvp1rqNbqvb6fa6vT4T3kF31J10Z91Fd9XddDfdXffQPXUv3Vv30X11P91fD9AD9SA9WA/RQ/UwPVyP0CP1KD1aj9FjIzNOj4/MBD0xMpP05MhM0VP/M9P0dD1Dz4zMLD07MnP03MjM0/P1Ar1QL9SL9GK9RC/Vy/RyvUKv1Kv0ar1Gr9Xr9Hq9QW/Um/RmvUVv1dv0dr1D79S79G69R+/V+/R+fUAf1If0YX1EH9XH9HF9Qp/Up/RpfUaf1ef0eX1BX9SX9GV9RV/V1/R1fUPf1Lf0bX1H39X39H39QD/Uj/Rj/UQ/1c/0c/1Cv9Sv9Gv9Rr/V7/R7/UF/1J/0Z/1Ff9Xf9Hf9Q//Uv/RvHWSCTYgJNWEmYMKNZ5DBBgwx1DDDjTDSKKONMdY445soJqqJZqKbGCamiWVimzgmroln4psEJqFJZBKbJCapSWaSmxQmpUllUps0Jq1JZ9KbDCajyWQymywmq8lmspscJqfJZXKbPCavyWfymwKmoClk8gQXNkVMUVPMFDclTElTypQ2ZUxZU86UNxVMRVPJVDZVTFVTzVQ3NUxNU8vUNnVMXRNh6pn6plbYPxqYdSF/6n/kn7P/tf7pWPkz/tMF28D891Xngv/crxV2Nvg/xv/Mzf/vE/5ZtS7kn1n/9cqf2tA0Mo1NE9PUNDMnw06GNTctTEvTyrQ2bUwbsyGkrWln2psOpqPpZDqbLqar6Wa6mx6mh+lpepnepo/pa/qafqa/GWAGmiBvkBlshphgb6gZZoabESbEG2lGmdFmjBlrQr1xZryZYCaaSWaymWLCvKlmmpluZpiZZpaZbeaYuWaemW8WmBTeQrPILDZLzFKzzCw3K8xKs8qsNmvMWrPOrDcbzEazyWw2W8xWs81sNzvMTrPL7DZ7zF6zz+w3+80Bc8AcNAfNIXPYHDFZg7MGHzXHzHFzwpw0p8xpc8acNefMeXPBXDSXzGVzxVw118x1c8PcNLfMbXPH3DX3zH3zwDw0j8xj88QUCykW8tQ8M8/Nc/PCvDAvzSvz2rwxb8078958MB/MR1M85I9P5rP5Yr6ab+a7+WF+ml/mtwmywTbEhthQG/avgA23nvVs9uDswchiCxYssdQyy62w0iqrrbHWOuvbKDaqjQiKZqPbGDamjWVj2zg2ro1n49suYQlsApvQJvYS2cQ2sU1ik9qkNplNblPYFDalTWVT2zQ2rU1n09sMNqP98yaPCMpkM9ssNqvNZrPb7DaHzWlz2dw2j81r89n8toAtaAvZwraILWKL2mK2mC1ui9sStqQtZUvbMrasLWfL2wq2oq1kK9sqtqqtZqvbGramrWVr2zq2ro2w9Wx928A2tI1sY9vENrXNbHPbwra0rWxr28a2te1se9vBdrSdbGfbxXa13Wx328P2tL1sb9vH9rX9bH87wA60Wb1BdpAdbIfYoXaYHW5H2JF2pB1lR9tzwWPsWJvDG2fH25zePybYiXaSnWyn2Kl2mp1uZ9iZdpadbefYuXaenW8X2IV2kV1sl9ildpldblfYlXaVXW3X2LW2kLfOrrcb7Ea7yW62W+xWu80OD9tud9iddpfdbffYvXaf3W8P2IP2kD1sj9ij9pg9bk/YEWEn7Sl72p6xZ+05e95esBftJXvZXrZX7FV7zV63N2yD4Jv2lr1t79i79p69bx/Yh/aRfWyf2Kf2mX1u94a8sC/tK/vavrFv7X90dL2z7+0H+9F+sp/tF/vVfrPf7Q/70/6yv22QC3YhLtSFuYALd55DDruoQeD+9HURR//FHHfCSVcntKk3IEi5P5p5f2j3KehTkHEzwqxzzndRXFQXzUV3MVxMF8vFdnFcXBfPxXcJXEKXyCV2SVxSl8zVC03uUkRK6VK51C6N6+CldelcehcRlMFldJlcZpfZZXFZXTaX3WV3OVxOl9PlcrlcbpfH5XX5XH5XwBV0hVxhV8QVdcVccVfClXSlXGlXxpV15Vx5V8FVdJVcZVfFVXXVXHVXw9V0tVxtV8f96RFrEVrXRbh6rr5r4Bq6Rq6xa+KaumauuWvhWrpWrrVr49q6dq696+A6uk6us+viurpurrvr4Xq6Xq636+P6un6uvxvgBrpBbrAb4oa6YW64G+FGulFutBvjxrpxbryb4Ca6SW6ym+KmumluupvhZrpZbrab4+a6eW6+W+AWukVusVvilrplbrlb4Va6VW61W+PWunVuvdvgNrpNbrPb4ra6bW672+F2ul1ut9vj9rp9br874A66Q+6wO+KOumPuuDvhTrpT7rQ74866c+68u+Auukvusrvirrpr7rq74W66W+62u+PuunvuvnvgHrpH7rF74p66Z+65e+FeulfutXvj3rp37r374D66T+6z++K+um/uu/vhfrpf7rcL8oP9ED/UD/MDfrjv+cjHPvjEpz7zuS986Stf+8a3vvN9P4of1Y/mR/dj+DH9WH5sP44f14/nx/cT+An9RH5iP4mf1E/mJ/dT+Cn9VH5qP42f1k/np/cz+Bn9TH5mP4uf1c/mZ/dz+Dn9XH5uP4+f18/n5/cL+AX9Qn5hv4hf1C/mF/dL+CX9Un5pv4xf1i/nl/cr+BX9Sn5lv4pf1a/mV/dr+DX9Wn5tv45f14/w6/n1/QZ+Q7+R39hv4jf1m/nN/RZ+S7+V39pv47f12/nt/Q5+R7+T39nv4nf1u/nd/d//16OH38Pv6ffye/t9/nPm/wDeBYjaJV4DAA=="
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
var LIBLLAMA_VERSION = "b12093-89026ef";
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
